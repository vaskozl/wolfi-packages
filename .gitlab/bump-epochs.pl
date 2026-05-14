#!/usr/bin/env perl
# bump-epochs — apply Epoch Rules (AGENTS.md) to yamls changed vs $BASE_REF:
#   * version bumped       → reset epoch to 0
#   * content edit, no bump → epoch += 1
#   * any rebuild trigger  → epoch += 1 on every reverse-dep
#
# --check fails on violations instead of fixing (used by CI). Default
# applies fixes in place.
#
# $BASE_REF defaults to HEAD~1. Renovate sets HEAD (pre-commit working tree);
# CI sets origin/main (full MR).
#
# Requires: melange, git, perl-yaml-pp.

use strict;
use warnings;
use feature 'say';
use Getopt::Long qw(:config bundling no_ignore_case);
use File::Basename qw(dirname);
use Cwd qw(abs_path);
use YAML::PP;

$| = 1;

my $check_mode = 0;
my $base_ref   = $ENV{BASE_REF} // 'HEAD~1';

GetOptions(
    'check'      => \$check_mode,
    'apply'      => sub { $check_mode = 0 },
    'base-ref=s' => \$base_ref,
    'help|h'     => sub { print_help(); exit 0 },
) or die "bad args (see --help)\n";

chdir abs_path(dirname(__FILE__) . '/..') or die "chdir: $!";

my $ypp = YAML::PP->new;

# --- helpers ----------------------------------------------------------------

sub run_capture {
    open my $fh, '-|', @_ or die "exec @_: $!";
    local $/;
    my $out = <$fh>;
    close $fh;
    chomp $out if defined $out;
    return $out // '';
}

sub slurp {
    open my $fh, '<', $_[0] or die "read $_[0]: $!";
    local $/;
    return scalar <$fh>;
}

sub git_blob {
    # File content at $ref, or undef if missing. close() returns false
    # when the child exits non-zero, which is how we detect absence.
    my ($ref, $file) = @_;
    my $pid = open my $fh, '-|';
    return undef unless defined $pid;
    if (!$pid) {
        open STDERR, '>', '/dev/null';
        exec 'git', 'show', "${ref}:${file}" or exit 127;
    }
    local $/;
    my $out = <$fh>;
    return close($fh) ? $out : undef;
}

sub run_capture_lines {
    open my $fh, '-|', @_ or die "exec @_: $!";
    my @lines = <$fh>;
    close $fh;
    chomp @lines;
    return @lines;
}

sub pkg_state {
    my ($yaml) = @_;
    my $doc = eval { $ypp->load_string($yaml) };
    return ('', 0) unless $doc && ref($doc) eq 'HASH' && $doc->{package};
    my $p = $doc->{package};
    return ($p->{version} // '', $p->{epoch} // 0);
}

sub rewrite_epoch {
    # Set epoch to $new_ep with optional new comment, dropping any existing one.
    my ($file, $new_ep, $reason) = @_;
    my $content = slurp($file);
    my $orig    = $content;
    my $tail    = (defined $reason && length $reason) ? " # $reason" : '';
    $content =~ s|^(\s*epoch:)[ \t]+\d+(?:[ \t]+\#[^\n]*)?$|$1 $new_ep$tail|m;
    return 0 if $content eq $orig;
    open my $wh, '>', $file or die "write $file: $!";
    print $wh $content;
    close $wh;
    return 1;
}

sub bump_epoch_keep_comment {
    # epoch += 1, keep any existing trailing comment (we can't infer a new one).
    my ($file) = @_;
    my $content = slurp($file);
    my $orig    = $content;
    my $new;
    $content =~ s|^(\s*epoch:)[ \t]+(\d+)([ \t]+\#[^\n]*)?$|
                  $new = $2 + 1;
                  "$1 $new" . ($3 // "")|me;
    return undef if $content eq $orig;
    open my $wh, '>', $file or die "write $file: $!";
    print $wh $content;
    close $wh;
    return $new;
}

sub print_help {
    open my $fh, '<', $0 or die $!;
    <$fh>;  # skip shebang
    while (<$fh>) {
        last unless /^#/ || /^\s*$/;
        s/^# ?//;
        print;
    }
}

# --- scan changes vs $base_ref ----------------------------------------------
# Top-level only — pipelines/*.yaml are shared snippets, not packages.

my @candidates = grep { !m{/} && -f $_ }
    run_capture_lines('git', 'diff', '--name-only', $base_ref, '--', '*.yaml');

if (!@candidates) {
    say "bump-epochs: no yaml changes vs $base_ref";
    exit 0;
}

my @content_no_bump;    # edited without version/epoch bump
my @bumped_bases;       # version or epoch differs
my %trig_ver;

for my $y (@candidates) {
    (my $base = $y) =~ s/\.yaml$//;
    my $blob = git_blob($base_ref, $y);
    next unless defined $blob;          # new file
    my $head = slurp($y);
    next if $head eq $blob;

    my ($cv, $ce) = pkg_state($head);
    my ($mv, $me) = pkg_state($blob);
    my $vc = $cv ne $mv;
    my $ec = "$ce" ne "$me";

    if ($vc || $ec) {
        push @bumped_bases, $base;
        $trig_ver{$base} = $cv;
    } else {
        push @content_no_bump, $y;
    }
}

if (!@content_no_bump && !@bumped_bases) {
    say "bump-epochs: no yaml changes vs $base_ref";
    exit 0;
}

my $fails = 0;
my $bumped = 0;

# Content edit with no bump: auto-bump in apply mode, fail in check mode.
if ($check_mode) {
    for my $f (@content_no_bump) {
        say "FAIL: $f changed but version and epoch unchanged — bump one";
        $fails++;
    }
} else {
    for my $f (@content_no_bump) {
        my ($cv, $ce) = pkg_state(slurp($f));
        my $new = bump_epoch_keep_comment($f);
        if (defined $new) {
            say "bumped $f: epoch $ce → $new (content change)";
            (my $base = $f) =~ s/\.yaml$//;
            push @bumped_bases, $base;
            $trig_ver{$base} = $cv;
            $bumped++;
        } else {
            say STDERR "WARN: $f has no 'epoch:' line — cannot auto-bump";
        }
    }
}

# Reset epoch to 0 on every version bump (unconditional, per AGENTS.md).
if (!$check_mode) {
    for my $base (@bumped_bases) {
        my $f    = "${base}.yaml";
        my $blob = git_blob($base_ref, $f);
        next unless defined $blob;
        my ($cv, $ce) = pkg_state(slurp($f));
        my ($mv, $me) = pkg_state($blob);
        next unless $cv ne $mv;
        next if $ce == 0;

        if (rewrite_epoch($f, 0, undef)) {
            say "reset $f: epoch $ce → 0 (version $mv → $cv)";
            $bumped++;
        }
    }
}

# --- DAG (only built when we have something to propagate) ------------------

opendir(my $dh, '.') or die "opendir .: $!";
my @yamls = sort grep { /\.yaml$/ && -f $_ } readdir $dh;
closedir $dh;

# Templated subpackage names (${{package.name}}-dev etc.) need melange to resolve.
my $QUERY = '{{ .Package.Name }}|{{ range $i,$s := .Subpackages }}{{ if $i }},{{ end }}{{ $s.Name }}{{ end }}|{{ range $i,$d := .Environment.Contents.Packages }}{{ if $i }},{{ end }}{{ $d }}{{ end }}{{ range .Package.Dependencies.Runtime }},{{ . }}{{ end }}';

my %provides;       # provided-name => [base, ...]
my %deps_of;

for my $y (@yamls) {
    (my $base = $y) =~ s/\.yaml$//;
    my $row = run_capture('melange', 'query', $y, $QUERY);
    my ($name, $subs, $deplist) = split /\|/, $row, 3;
    push @{ $provides{$name} }, $base if defined $name && length $name;
    for my $s (grep { length } split /,/, ($subs // '')) {
        push @{ $provides{$s} }, $base;
    }
    my @ds;
    for my $d (split /,/, ($deplist // '')) {
        $d =~ s/[~=<>!].*//;
        push @ds, $d if length $d;
    }
    $deps_of{$base} = \@ds;
}

my %rev;
for my $base (sort keys %deps_of) {
    for my $name (@{ $deps_of{$base} }) {
        next unless exists $provides{$name};
        for my $db (@{ $provides{$name} }) {
            next if $db eq $base;
            push @{ $rev{$db} }, $base;
        }
    }
}
for my $k (keys %rev) {
    my %u;
    $rev{$k} = [ sort grep { !$u{$_}++ } @{ $rev{$k} } ];
}

# --- transitive reverse-dep closure ----------------------------------------
# Re-read on-disk state so the epoch-reset above is reflected.
my %changed_set = map { $_ => 1 } @bumped_bases;
my %effective_triggers;
for my $base (@bumped_bases) {
    my $f    = "${base}.yaml";
    my $blob = git_blob($base_ref, $f) // '';
    my ($cv, $ce) = pkg_state(slurp($f));
    my ($mv, $me) = pkg_state($blob);
    if ($cv ne $mv || "$ce" ne "$me") {
        $effective_triggers{$base} = $cv;
    }
}

my %first_trigger;
for my $trig (sort keys %effective_triggers) {
    my @q = ($trig);
    my %seen;
    while (@q) {
        my $cur = shift @q;
        next if $seen{$cur}++;
        $first_trigger{$cur} //= $trig if $cur ne $trig;
        push @q, @{ $rev{$cur} // [] };
    }
}

my @required = sort grep { !$changed_set{$_} } keys %first_trigger;

for my $rd (@required) {
    my $f = "${rd}.yaml";
    unless (-f $f) {
        say "skip $rd: $f missing (deleted in this branch?)";
        next;
    }

    my $blob = git_blob($base_ref, $f);
    my ($cv, $ce) = pkg_state(slurp($f));
    my ($mv, $me) = defined $blob ? pkg_state($blob) : ('', 0);
    next if $cv ne $mv || "$ce" ne "$me";   # already bumped

    my $trig   = $first_trigger{$rd};
    my $tver   = $effective_triggers{$trig} // '';
    my $reason = "rebuild for $trig" . (length $tver ? " $tver" : '');

    if ($check_mode) {
        say "MISSING: $f needs epoch bump ($reason)";
        $fails++;
        next;
    }

    my $new_ep = $ce + 1;
    if (rewrite_epoch($f, $new_ep, $reason)) {
        say "bumped $f: epoch $ce → $new_ep # $reason";
        $bumped++;
    } else {
        say STDERR "FAIL: $f has no 'epoch:' line — cannot bump";
        $fails++;
    }
}

if ($check_mode && $fails) {
    say STDERR "";
    say STDERR "$fails problem(s) found vs $base_ref.";
    say STDERR "Run `just bump-epochs` (or `.gitlab/bump-epochs.pl --apply`) to fix.";
    exit 1;
}

unless ($check_mode) {
    say "bump-epochs: $bumped epoch bump(s); $fails failure(s)";
    exit($fails ? 1 : 0);
}
