#!/bin/sh
dd bs=1 count=1 of=/dev/null 2>/dev/null
dd if=/dev/zero bs=4096 count=4 2>/dev/null
cat >/dev/null
