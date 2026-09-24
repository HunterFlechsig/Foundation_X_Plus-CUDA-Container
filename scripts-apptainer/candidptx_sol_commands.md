# CANDID-PTX SOL commands (50 cycles)

Run these on SOL from the repo checkout on branch `candidptx-tasksets`.
Do not rebuild split files. Do not copy the official zip.

Each selected task is trained 50 times. The training loop stops before
`total_epochs`, so the values are 51 (a–c), 101 (d–f), and 151 (g).
Jobs use two A100s, a 7-day wall, account `grp_jliang12`, partition
`public`, QoS `public`. If a job hits the wall, resume it from its last
checkpoint with the same `TOTAL_EPOCHS`.

Outputs stay under `/scratch/hflechsi/FoundationX/candidptx_tasksets/`.

## 0. Update the checkout

```bash
cd ~/path/to/Foundation_X_Plus-CUDA-Container
git fetch origin
git checkout candidptx-tasksets
git pull origin candidptx-tasksets
```

Confirm `--find_unused_params` is on the launcher line, and that
`main_Consolidated.py` does **not** assign `zeros_like` onto frozen
gradients before `backward`.

## 1. Smoke of task set (g), one cycle

```bash
interactive -p htc -q public -A grp_jliang12 -G a100:2 -c 10 --mem=100G -t 0-4
```

Inside that allocation:

```bash
cd ~/path/to/Foundation_X_Plus-CUDA-Container
./scripts-apptainer/run_apptainer_candidptx_taskset.sh smoke
```

If `htc` is too short for localization, use public instead:

```bash
interactive -p public -q public -A grp_jliang12 --gres=gpu:a100:2 -c 10 --mem=100G -t 0-12
```

Success is all of:

```bash
ls /scratch/hflechsi/FoundationX/candidptx_tasksets/smoke/ckpt_E1_TH9.pth \
   /scratch/hflechsi/FoundationX/candidptx_tasksets/smoke/ckpt_E2_TH10.pth \
   /scratch/hflechsi/FoundationX/candidptx_tasksets/smoke/ckpt_E3_TH11.pth
```

and Student plus Teacher rows for classification, localization, and
segmentation in `export_csvFile.csv` after epoch 1. Then `exit` the
interactive session.

Do not start f, g, or the a–c resumes until that smoke finishes.

## 2. Move the failed f and g directories

```bash
cd /scratch/hflechsi/FoundationX/candidptx_tasksets
mv f f_failed_loc
mv g g_failed_loc
```

## 3. Start f and g from epoch 1

From the repo checkout:

```bash
cd ~/path/to/Foundation_X_Plus-CUDA-Container

sbatch --job-name=candidptx_f --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=101 \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh f

sbatch --job-name=candidptx_g --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=151 \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh g
```

Or:

```bash
./scripts-apptainer/submit_candidptx_tasksets.sh
```

## 4. Resume a, b, and c (10-cycle jobs already finished)

Print the last checkpoint, then submit. Replace the `RESUME` path if
`ls` shows a different file.

```bash
cd ~/path/to/Foundation_X_Plus-CUDA-Container
OUT=/scratch/hflechsi/FoundationX/candidptx_tasksets

ls -1 "$OUT/a"/ckpt_E*_TH*.pth | sort -V | tail -1
ls -1 "$OUT/b"/ckpt_E*_TH*.pth | sort -V | tail -1
ls -1 "$OUT/c"/ckpt_E*_TH*.pth | sort -V | tail -1
```

```bash
sbatch --job-name=candidptx_a --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=51,RESUME=$(ls -1 "$OUT/a"/ckpt_E*_TH*.pth | sort -V | tail -1) \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh a

sbatch --job-name=candidptx_b --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=51,RESUME=$(ls -1 "$OUT/b"/ckpt_E*_TH*.pth | sort -V | tail -1) \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh b

sbatch --job-name=candidptx_c --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=51,RESUME=$(ls -1 "$OUT/c"/ckpt_E*_TH*.pth | sort -V | tail -1) \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh c
```

These keep writing into the existing `a`, `b`, and `c` directories.

## 5. Resume d and e after their 10-cycle jobs finish

Check that those jobs are no longer in the queue:

```bash
squeue -u "$USER" -n candidptx_d,candidptx_e
```

Then:

```bash
cd ~/path/to/Foundation_X_Plus-CUDA-Container
OUT=/scratch/hflechsi/FoundationX/candidptx_tasksets

ls -1 "$OUT/d"/ckpt_E*_TH*.pth | sort -V | tail -1
ls -1 "$OUT/e"/ckpt_E*_TH*.pth | sort -V | tail -1

sbatch --job-name=candidptx_d --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=101,RESUME=$(ls -1 "$OUT/d"/ckpt_E*_TH*.pth | sort -V | tail -1) \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh d

sbatch --job-name=candidptx_e --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=101,RESUME=$(ls -1 "$OUT/e"/ckpt_E*_TH*.pth | sort -V | tail -1) \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh e
```

## 6. If a 7-day job stops at the wall

Same command as the resume that started it, with `RESUME` pointed at the
new last checkpoint. Keep the same `TOTAL_EPOCHS`.

```bash
OUT=/scratch/hflechsi/FoundationX/candidptx_tasksets
NAME=g   # a b c d e f or g
TOTAL=151   # 51 for a–c, 101 for d–f, 151 for g

sbatch --job-name=candidptx_${NAME} --time=7-00:00:00 -A grp_jliang12 -p public -q public \
  --gres=gpu:a100:2 --cpus-per-task=10 --mem=100G \
  --export=ALL,TOTAL_EPOCHS=$TOTAL,RESUME=$(ls -1 "$OUT/$NAME"/ckpt_E*_TH*.pth | sort -V | tail -1) \
  scripts-apptainer/run_apptainer_candidptx_taskset.sh "$NAME"
```

Slurm stdout/stderr land in the directory where you ran `sbatch`, as
`candidptx_<name>_<jobid>.out` and `.err`.
