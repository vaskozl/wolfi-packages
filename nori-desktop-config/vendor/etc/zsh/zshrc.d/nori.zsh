PROMPT='%m:%~%# '

export FORGIT_FZF_DEFAULT_OPTS='--height=100% --preview-window=up:50% --layout=reverse-list'

for i in /usr/share/zsh/plugins/*/*.zsh; do . "$i"; done

alias vi=nvim
alias vim=nvim
alias claude='podman run --security-opt unmask=ALL --userns=keep-id:uid=568 -it --net=host --rm -w $PWD -v $HOME -v $HOME/.claude.json:$HOME/.claude.json -v $HOME/.claude:$HOME/.claude -e HOME=$HOME -v $PWD:$PWD ghcr.io/vaskozl/claude --dangerously-skip-permissions'
alias codex='podman run -it --net=host --userns=keep-id:uid=568 --rm -w $PWD -v $HOME -v $HOME/.codex:$HOME/.codex -e HOME=$HOME -v $PWD:$PWD ghcr.io/vaskozl/codex --yolo'

unset MAIL MAILPATH

fastfetch

source <(fzf --zsh)
