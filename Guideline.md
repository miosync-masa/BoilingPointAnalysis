# MD実行ガードシステム - 再実行ゼロ運用
## もう二度とやり直しをさせないための完全防御システム

---

## 0. 原則（止めるための3ルール）

1. **ゴールデン設定を固定**（モデル×系ごとに1枚）：`TIP3P_slab.mdp.gold` 等
2. 実行前に**Preflight**（設定自動チェック）を必ず通す
3. 設定を変えるときは**新バージョン**を作る（上書き・途中差し替え禁止）

---

## 1. Preflight（実行前ガード）

### A. MDPガード（必須キー＆値の一致チェック）

`mdp_guard.sh`（プロジェクト直下に置く）

```bash
#!/usr/bin/env bash
set -euo pipefail

GOLD="$1"      # 例: mdp_gold/TIP3P_slab_4ns.mdp.gold
CAND="$2"      # 実行予定の .mdp

req=(
 "integrator=md"
 "dt=0.002"
 "nsteps=2000000"
 "tcoupl=nose-hoover"
 "tau-t=0.5"
 "cutoff-scheme=Verlet"
 "rlist=1.2"
 "rcoulomb=1.2"
 "rvdw=1.2"
 "coulombtype=PME"
 "pme-order=4"
 "fourierspacing=0.12"
 "DispCorr=EnerPres"
 "ewald-geometry=3dc"
 "epsilon-surface=0"
 "pcoupl=no"
)

miss=0
for kv in "${req[@]}"; do
  k="${kv%%=*}"; v="${kv#*=}"
  gv=$(grep -E "^[[:space:]]*${k}[[:space:]]*=" "$CAND" | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2);print $2}')
  if [[ "$gv" != "$v" ]]; then
    echo "[NG] $k = $gv  (need $v)"
    miss=1
  else
    echo "[OK] $k = $gv"
  fi
done

# constraints はモデルで条件分岐（TIP3P/SPC/E: h-bonds, TIP4P/2005: all-bonds）
model_hint=$(basename "$CAND" | tr '[:upper:]' '[:lower:]')
if grep -qi "tip4p" <<< "$model_hint"; then
  need="all-bonds"
else
  need="h-bonds"
fi
cv=$(grep -E "^[[:space:]]*constraints[[:space:]]*=" "$CAND" | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2);print $2}')
if [[ "$cv" != "$need" ]]; then
  echo "[NG] constraints = $cv (need $need)"; miss=1
else
  echo "[OK] constraints = $cv"
fi

# 出力頻度（解析分解能の確保）
for k in nstenergy nstxout-compressed nstlog; do
  [[ "$k" == "nstenergy" ]] && want=100 || want=2000
  gv=$(grep -E "^[[:space:]]*${k}[[:space:]]*=" "$CAND" | tail -1 | awk -F= '{gsub(/[[:space:]]/,"",$2);print $2}')
  if [[ "$gv" != "$want" ]]; then
    echo "[NG] $k = $gv (need $want)"; miss=1
  else
    echo "[OK] $k = $gv"
  fi
done

if [[ $miss -ne 0 ]]; then
  echo "==> Preflight FAILED"; exit 2
else
  echo "==> Preflight PASSED"
fi
```

### B. gromppドライラン（未知キー/表記ミス潰し）

```bash
gmx grompp -f nvt_slab_100C.mdp -c conf.gro -p topol.top -o dry.tpr -maxwarn 0
```

※1つでもWarning/Unknownが出たら走らせない

---

## 2. ゴールデンMDPテンプレート

### TIP3P/SPC/E用（バルク＆スラブ共通ベース）

```mdp
; === 実行パラメータ ===
integrator              = md
dt                      = 0.002
nsteps                  = 2000000        ; 4 ns

; === 出力制御 ===
nstenergy               = 100            ; 0.2 ps ごと（α計算用）
nstxout-compressed      = 2000           ; 4 ps ごと
nstvout                 = 0
nstfout                 = 0
nstlog                  = 2000

; === 温度制御 ===
tcoupl                  = nose-hoover
tc-grps                 = System
tau-t                   = 0.5
ref-t                   = __REF_T__      ; sedで置換

; === 初期速度生成 ===
gen-vel                 = yes
gen-temp                = __REF_T__
gen-seed                = -1             ; レプリカ独立性

; === 相互作用計算 ===
cutoff-scheme           = Verlet
nstlist                 = 10
rlist                   = 1.2
coulombtype             = PME
rcoulomb                = 1.2
pme-order               = 4
fourierspacing          = 0.12

; === vdW＆長距離補正 ===
vdwtype                 = Cut-off
rvdw                    = 1.2
DispCorr                = EnerPres       ; 必須！

; === 拘束条件 ===
constraints             = h-bonds        ; TIP3P/SPC/E用
constraint-algorithm    = LINCS
lincs-iter              = 2
lincs-order             = 4

; === PBC ===
pbc                     = xyz
comm-mode               = Linear

; === スラブ系追加設定（バルクでは無視される）===
ewald-geometry          = 3dc            ; スラブ用
epsilon-surface         = 0              ; スラブ用
pcoupl                  = no              ; NVT
```

### TIP4P/2005用（constraints違い）

```mdp
; TIP4P/2005は上記と同じだが、以下を変更：
constraints             = all-bonds      ; TIP4P/2005用
```

---

## 3. 実行テンプレート

`run_batch.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
MODEL=${1:?tip3p|spce|tip4p2005}
TEMPS=${2:-"Tb-2 Tb Tb+1"}  # 例: "98 100 102" / "109 110 111"
BASE=~/lambda3_boiling_proof/organized/journal_submission

TPL=$BASE/mdp_gold/${MODEL}_slab_4ns.mdp.gold

for T in $TEMPS; do
  K=$(python3 -c "print($T + 273.15)")
  dir=$BASE/slab_nvt/${MODEL}/${T}C
  mkdir -p "$dir"
  cp -n $BASE/common/conf_slab_${MODEL}.gro "$dir"/conf.gro || true
  cp -n $BASE/common/topol_${MODEL}.top "$dir"/topol.top || true

  for rep in 1 2 3 4 5; do
    R=$dir/rep${rep}
    mkdir -p "$R"; cd "$R"
    
    # mdp生成
    sed "s/__REF_T__/${K}/g" "$TPL" > nvt_${T}C_rep${rep}.mdp
    
    # preflight
    bash $BASE/mdp_guard.sh "$TPL" nvt_${T}C_rep${rep}.mdp
    
    # grompp
    gmx grompp -f nvt_${T}C_rep${rep}.mdp -c "$dir"/conf.gro -p "$dir"/topol.top -o nvt_${T}C_rep${rep}.tpr
    
    # 実行（再開安全）
    nohup gmx mdrun -deffnm nvt_${T}C_rep${rep} -cpt 10 -append -maxh 23 > run.log 2>&1 &
    cd - >/dev/null
    sleep 1
  done
done
```

---

## 4. Postflight（実行後チェック）

`postflight_check.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
E=$1  # .edr
XTC=${2:-}  # .xtc (任意)

# 必須エネルギー項目
must=("Kinetic-En." "Vir-XX" "Vir-YY" "Vir-ZZ" "Pressure-XX" "Pressure-YY" "Pressure-ZZ" "Volume")
out=$(echo -e "?" | gmx energy -f "$E" 2>/dev/null || true)
fail=0
for k in "${must[@]}"; do
  grep -q "$k" <<< "$out" || { echo "[NG] missing: $k"; fail=1; }
done

# 走行時間（4 ns 相当）
n=$(gmx check -f "$XTC" 2>/dev/null | awk '/Step/{print $2}' | tail -1)
[[ -n "$n" && "$n" -ge 2000000 ]] || { echo "[NG] frames/steps too short"; fail=1; }

# 温度の実現（±1 K）
avgT=$(echo "Temperature" | gmx energy -f "$E" 2>/dev/null | awk '/^Average/{print $2}' )
if [[ -n "$avgT" && -n "$REF_T" ]]; then
  awk -v t="$avgT" -v r="$REF_T" 'BEGIN{if (t<r-1 || t>r+1) exit 1}' || { echo "[NG] Temperature off: $avgT vs $REF_T"; fail=1; }
fi

[[ $fail -eq 0 ]] && echo "==> Postflight PASSED" || { echo "==> Postflight FAILED"; exit 2; }
```

---

## 5. データ管理構造

```
/mdp_gold/
  TIP3P_slab_4ns.mdp.gold
  SPCE_slab_4ns.mdp.gold
  TIP4P2005_slab_4ns.mdp.gold
  TIP3P_bulk_500ps.mdp.gold
  SPCE_bulk_500ps.mdp.gold
  TIP4P2005_bulk_500ps.mdp.gold

/common/
  conf_slab_tip3p.gro    topol_tip3p.top
  conf_slab_spce.gro     topol_spce.top
  conf_slab_tip4p.gro    topol_tip4p2005.top
  conf_bulk_tip3p.gro
  conf_bulk_spce.gro
  conf_bulk_tip4p.gro

/slab_nvt/<model>/<temp>C/rep#/   (mdp, tpr, edr, xtc, log)
/bulk_nvt/<model>/<temp>C/rep#/   (mdp, tpr, edr, xtc, log)

RUN_MANIFEST.yaml      （モデル/温度/seed/MD5/コマンドを記録）
CHANGELOG.md          （設定差分は追記のみ）
```

---

## 6. 最終チェックリスト（声出し確認）

- [ ] 対象モデルの **gold mdp** から生成した？
- [ ] **mdp_guard** と **grompp ドライラン**を通過？
- [ ] `DispCorr=EnerPres` 入ってる？（超重要！）
- [ ] スラブなら `ewald-geometry=3dc` と `epsilon-surface=0`？
- [ ] `nstenergy=100` / `nstxout-compressed=2000`？
- [ ] constraints（TIP3P/SPC=**h-bonds**、TIP4P/2005=**all-bonds**）正しい？
- [ ] Lzは十分厚い？（スラブなら15-20 nm）
- [ ] `gen-seed=-1` でレプリカ独立？
- [ ] MANIFEST に記録した？

---

## 7. 実行コマンド例（Frozen Spec v1.0 確定版）

### スラブ系（NVT, 4ns, 5レプリカ/点）

```bash
# TIP3P - 5温度×5レプリカ = 25本
MODEL=tip3p
TEMPS="95 98 100 102 105"
bash run_batch.sh $MODEL "$TEMPS"

# SPC/E - 5温度×5レプリカ = 25本
MODEL=spce
TEMPS="105 108 109 110 111"
bash run_batch.sh $MODEL "$TEMPS"

# TIP4P/2005 - 5温度×5レプリカ = 25本
MODEL=tip4p2005
TEMPS="97 99 100 101 103"
bash run_batch.sh $MODEL "$TEMPS"
```

### バルク系（NVT, 1ns, 5レプリカ/点）

```bash
# TIP3P
MODEL=tip3p
TEMPS="95 98 100 102 105"
bash run_batch.sh $MODEL "$TEMPS"

# SPC/E
MODEL=spce
TEMPS="105 108 109 110 111"
bash run_batch.sh $MODEL "$TEMPS"

# TIP4P/2005
MODEL=tip4p2005
TEMPS="97 99 100 101 103"
bash run_batch.sh $MODEL "$TEMPS"
```

**合計：150本のMDシミュレーション**（スラブ75本＋バルク75本）

※温度点は削らない・フルで回す・一切ブレない（Frozen Spec v1.0）

---

## まとめ

**ゴールデン → Preflight → Run → Postflight**

この4段階で「運用で守る」。もう二度とやり直しなんてさせない。

「現れるべきところに現れる、だからこそ物理」

---

*最終更新: 2025年9月8日*  
*二度とやり直しをさせないために*
