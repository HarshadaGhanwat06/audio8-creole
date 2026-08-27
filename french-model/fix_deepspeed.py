#!/usr/bin/env python3
"""
Fix audio8_tts_sft.sh - Disable DeepSpeed for single-GPU training.
Run this on the GPU host.
"""
import shutil
import re
import subprocess
import sys

SFT = "/home/ubuntu/audio8/Audio8_TTS/audio8_tts_sft.sh"
BAK = SFT + ".bak"

def main():
    print("=" * 50)
    print("Fixing DeepSpeed in audio8_tts_sft.sh")
    print("=" * 50)

    # 1. Backup
    print("\n[1] Creating backup...")
    shutil.copy2(SFT, BAK)
    print(f"    Backup: {BAK}")

    # 2. Read file
    print("\n[2] Reading script...")
    with open(SFT, "r") as f:
        content = f.read()
    lines = content.split("\n")

    # 3. Show current DeepSpeed lines
    print("\n[3] Current DeepSpeed config:")
    for i, line in enumerate(lines, 1):
        if "DEEPSPEED" in line or "deepspeed" in line.lower():
            print(f"    L{i}: {line}")

    # 4. Fix DEEPSPEED_CONFIG default
    print("\n[4] Fixing DEEPSPEED_CONFIG default...")
    content = content.replace(
        'DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}"',
        'DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"'
    )
    print("    Changed to empty default")

    # 5. Remove the unconditional --deepspeed line
    print("\n[5] Removing unconditional --deepspeed...")
    new_lines = []
    for line in lines:
        if '--deepspeed "${DEEPSPEED_CONFIG:-${PROJECT_ROOT}/configs/deepspeed_zero2.json}"' in line:
            print(f"    Removed: {line.strip()}")
            continue
        new_lines.append(line)
    content = "\n".join(new_lines)

    # 6. Find the training command and make deepspeed conditional
    print("\n[6] Making DeepSpeed conditional...")
    lines = content.split("\n")

    # Find the python audio8_tts_sft.py line
    train_line_idx = None
    for i, line in enumerate(lines):
        if re.search(r"python.*audio8_tts_sft\.py", line):
            train_line_idx = i
            break

    if train_line_idx is not None:
        # Check if we already have DS_ARGS
        has_ds_args = any("DS_ARGS" in l for l in lines)

        if not has_ds_args:
            # Insert conditional block before the training command
            conditional = [
                "# DeepSpeed: only use if config is set",
                'DS_ARGS=""',
                'if [ -n "${DEEPSPEED_CONFIG}" ]; then',
                '    DS_ARGS="--deepspeed \\"${DEEPSPEED_CONFIG}\\""',
                "fi",
                "",
            ]
            lines = lines[:train_line_idx] + conditional + lines[train_line_idx:]

            # Find the training command again (now shifted)
            for i, line in enumerate(lines):
                if re.search(r"python.*audio8_tts_sft\.py", line):
                    train_line_idx = i
                    break

        # Add ${DS_ARGS} to the training command if not already there
        if "${DS_ARGS}" not in lines[train_line_idx] and "${DEEPSPEED_ARGS}" not in lines[train_line_idx]:
            # Append before trailing backslash
            line = lines[train_line_idx]
            if line.rstrip().endswith("\\"):
                lines[train_line_idx] = line.rstrip()[:-1] + " ${DS_ARGS} \\"
            else:
                lines[train_line_idx] = line + " ${DS_ARGS}"
            print("    Added ${DS_ARGS} to training command")

        content = "\n".join(lines)
    else:
        print("    WARNING: Could not find training command!")

    # 7. Write modified file
    print("\n[7] Writing modified script...")
    with open(SFT, "w") as f:
        f.write(content)
    print("    Done")

    # 8. Verify syntax
    print("\n[8] Verifying syntax...")
    result = subprocess.run(["bash", "-n", SFT], capture_output=True, text=True)
    if result.returncode == 0:
        print("    Syntax check: PASS")
    else:
        print(f"    Syntax check: FAIL")
        print(f"    Error: {result.stderr}")
        print("    Restoring backup...")
        shutil.copy2(BAK, SFT)
        sys.exit(1)

    # 9. Show modified DeepSpeed section
    print("\n[9] Modified DeepSpeed config:")
    with open(SFT, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if "DEEPSPEED" in line or "deepspeed" in line.lower() or "DS_ARGS" in line:
            print(f"    L{i}: {line.rstrip()}")

    print("\n" + "=" * 50)
    print("DONE! Script ready for single-GPU training.")
    print("=" * 50)

if __name__ == "__main__":
    main()
