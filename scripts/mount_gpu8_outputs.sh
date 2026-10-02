#!/usr/bin/env bash
set -euo pipefail
mount_dir=/home/fbohan/AIC/checkpoints/gpu8_storage
mkdir -p "$mount_dir"
if ! mountpoint -q "$mount_dir"; then
  /home/fbohan/.local/lib/aicomp/sshfs \
    -o IdentityFile=/home/fbohan/.ssh/aicomp_servers_ed25519,IdentitiesOnly=yes,BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    fbohan@222.20.97.217:/home/fbohan/AIC "$mount_dir"
fi
findmnt -n -T "$mount_dir" -o SOURCE,FSTYPE
