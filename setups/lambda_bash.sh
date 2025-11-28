#!/bin/bash

# Update SSH to look for ssh keys in local filestystem
ln -sf ~/ktaht/.ssh ~/.ssh

eval "$(ssh-agent -s)"
ssh-add ~/ktaht/.ssh/id_ed25519

# Note, to test run:
# ssh -T git@github.com

echo "✓ Setup Complete - Ready to Work!"

