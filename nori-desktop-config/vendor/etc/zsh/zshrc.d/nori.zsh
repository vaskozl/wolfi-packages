PROMPT='%m:%~%# '

for i in /usr/share/zsh/plugins/*/*.zsh; do . "$i"; done

alias vi=nvim
alias vim=nvim
alias claude='podman run --security-opt unmask=ALL --userns=keep-id:uid=568 -it --net=host --rm -w $PWD -v $HOME -v $HOME/.claude.json:$HOME/.claude.json -v $HOME/.claude:$HOME/.claude -e HOME=$HOME -v $PWD:$PWD ghcr.io/vaskozl/claude --dangerously-skip-permissions'
alias codex='podman run -it --net=host --userns=keep-id:uid=568 --rm -w $PWD -v $HOME -v $HOME/.codex:$HOME/.codex -e HOME=$HOME -v $PWD:$PWD ghcr.io/vaskozl/codex --yolo'
alias pi='sudo podman run -u $(id -u) -it --net=host --rm -w $PWD -v $HOME -v $HOME/.pi:$HOME/.pi -e HOME=$HOME -v $PWD:$PWD --entrypoint pi pi-coding-agent'

unset MAIL MAILPATH

fastfetch

source <(fzf --zsh)
