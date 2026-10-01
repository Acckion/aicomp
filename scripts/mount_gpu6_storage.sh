#!/usr/bin/env bash
# Mount our own GPU6 disk directory; no sudo or changes to other users.
set -euo pipefail
mount_dir=/home/fbohan/AIC/checkpoints/gpu6_storage
mkdir -p "$mount_dir"
if ! mountpoint -q "$mount_dir"; then
  /home/fbohan/.local/lib/aicomp/sshfs \
    -o IdentityFile=/home/fbohan/.ssh/aicomp_servers_ed25519,IdentitiesOnly=yes,BatchMode=yes,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 \
    fbohan@222.20.97.104:/home2/fbohan/AIC_storage "$mount_dir"
fi
df -h "$mount_dir"
