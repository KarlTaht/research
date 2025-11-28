#!/bin/bash


# Start ssh-agent only if not already running
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)"
fi

# Add key only if not already added
if ! ssh-add -l | grep -q "id_ed25519"; then
    ssh-add ~/ktaht/.ssh/id_ed25519
fi

# Note, to test run:
# ssh -T git@github.com

# Personal Preferences
export EDITOR=vim
export VISUAL=vim

alias ls='ls -lhG --group-directories-first --color=auto'
PS1='\[\033[01;30m\][\[\033[01;31m\]\u\[\033[01;37m\]@\[\033[01;37m\]\h\[\033[01;30m\]]\[\033[00m\]:\[\033[01;35m\]/\W\[\033[00m\]\$ '

# Install Claude Code if not present

# Install Claude Code only if --claude flag is passed
if [[ "$1" == "--claude" ]]; then
    if ! command -v claude &> /dev/null; then
        echo "Installing Claude Code..."
        curl -fsSL https://claude.ai/install.sh | sh
    else
        echo "Claude Code already installed"
    fi
fi

echo "✓ Setup Complete - Ready to Work!"

