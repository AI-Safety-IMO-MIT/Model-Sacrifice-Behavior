# RunPod + VSCode over SSH

Edit on Mac, run on the pod's GPU. No export/upload.

## The permanent fix: a custom template

RunPod only starts sshd if the env var `PUBLIC_KEY` exists in the pod.
It builds that var by concatenating EVERY key in the account's
"SSH Public Keys" list. This is a shared/team account with ~41 keys,
and the injection silently fails -- pods get no `PUBLIC_KEY` at all
(verified: `/etc/rp_environment` contains only `RUNPOD_PUBLIC_IP`).

Your key IS registered correctly (name `Sarunyu`, fingerprint
`SHA256:LsCRyFD0/uT65iRIldinzHic3edp5XuSoMF6bX+NQlI`). The account
list is just too big for RunPod to inject. Nothing you can fix from
the settings page without deleting other people's keys.

Putting `PUBLIC_KEY` in your own template is self-contained, needs no
account changes, and always works.

1. Console -> **Templates** -> **New Template**
2. Container image: the same one you normally use, e.g.
   `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
3. Expose **TCP port 22** (and 8888 if you want Jupyter)
4. **Environment Variables** -> Add:

   | Key | Value |
   |---|---|
   | `PUBLIC_KEY` | `ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBYQ4oVPU+aEMoJ166PyaSjwpnC7W4PeHlzT2mjkZvce sarunyu2005@gmail.com` |

5. Save. Always deploy from this template.

Every pod from it starts sshd automatically. No web terminal step.

## Mac side (already done, once ever)

Key has a passphrase, so load it into the agent + Keychain:

```
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

Survives reboots. Without this, VSCode fails with
"Permission denied (publickey)" even though the key is correct.

## Each new pod

1. Deploy from your template.
2. Wait for Logs -> `Start script(s) finished, Pod is ready to use.`
3. Connect tab -> copy the **SSH over exposed TCP** line
   (NOT the `@ssh.runpod.io` one -- that has no SFTP, so VSCode fails).
4. VSCode: `Cmd+Shift+P` -> **Remote-SSH: Add New SSH Host** ->
   paste the whole `ssh root@IP -p PORT -i ~/.ssh/id_ed25519` line.
5. Connect. First connect downloads ~213 MB of VSCode server.

## Check the machine before trusting it

Bad hosts exist. One pod ran at 20 KB/s -- the VSCode server download
would have taken 100 minutes. Test first:

```
curl -o /dev/null -s -w "%{speed_download}\n" -m 15 \
  "https://speed.cloudflare.com/__down?bytes=50000000"
```

- `> 5000000` (5 MB/s) -> fine
- `< 1000000` (1 MB/s) -> terminate, redeploy elsewhere

## Rescue a pod with no SSH

If sshd isn't running, enable the **web terminal** in the Connect tab:

```
mkdir -p ~/.ssh && echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBYQ4oVPU+aEMoJ166PyaSjwpnC7W4PeHlzT2mjkZvce sarunyu2005@gmail.com' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && ssh-keygen -A && service ssh start
```

## Diagnosing

| Symptom | Meaning |
|---|---|
| `Connection refused` | sshd not running -> no `PUBLIC_KEY` injected |
| `Permission denied (publickey)` | key not loaded on Mac -> run `ssh-add` |
| HTTP 502 on a proxy port | nothing listening on that port (often fine) |
| Stuck "Downloading VS Code Server" | slow machine -> speed-test, probably terminate |

Check what RunPod actually injected, from inside the pod:

```
cat /etc/rp_environment | grep -i public
```

`PUBLIC_KEY=...` present = good. Only `RUNPOD_PUBLIC_IP` = key missing.

## Gotchas

- Only `/workspace` survives a stop/restart, and only on a network volume.
- Resuming gives a **new IP and port** -> re-add the host in VSCode.
- Use `tmux` for long training runs so a dropped SSH doesn't kill the job.
- Get code onto the pod with `git clone`, not file copying.
