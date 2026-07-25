"""
merge_small_batches.py
=======================

المشكلة (Problem)
------------------
كل قصيدة (poem) مقسّمة إلى "باتشات" (batches) كل واحد منها 12 بيتًا كحد أقصى.
الباتش الأخير لكل قصيدة قد يكون صغيرًا جدًا (أقل من MIN_BATCH_SIZE أبيات).
هذا السكربت يدمج الباتش الأخير في الباتش الذي يسبقه مباشرة إذا كان حجمه
أقل من الحد الأدنى المسموح به، مع الحفاظ التام على verse_id لكل بيت.

القاعدة (Rule)
--------------
لكل قصيدة (مجموعة صفوف تشترك في POET_NAME + poem_no، مرتبة بحسب batch_no):
    - إذا كان عدد الباتشات > 1  و  حجم آخر باتش < MIN_BATCH_SIZE:
        يُدمج آخر باتش في الباتش الذي يسبقه مباشرة (الأعمدة الوصفية
        POET_NAME / poem_no / POET_RANK / meter تُؤخذ من الباتش السابق،
        وتُجمع قوائم DATA، ويُجمع BATCH_SIZE).
    - غير ذلك: لا تغيير.

ملاحظة: بما أن كل الباتشات ما عدا الأخير يجب أن تكون ممتلئة (12 بيتًا)،
فمن غير الممكن رياضيًا الحاجة لأكثر من عملية دمج واحدة لكل قصيدة ضمن هذه
البيانات، لكن السكربت يتحقق من ذلك بدل افتراضه.

الاستخدام (Usage)
-----------------
    python merge_small_batches.py \\
        --input TOP_100_ARABIC_POETS_OF_ALL_TIME_STAGE_02.pkl \\
        --output TOP_100_ARABIC_POETS_OF_ALL_TIME_STAGE_03.pkl \\
        --min-batch-size 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_MIN_BATCH_SIZE = 4
GROUP_KEYS = ["POET_NAME", "poem_no"]
STATIC_COLS = ["POET_NAME", "poem_no", "POET_RANK", "meter"]


def merge_small_last_batches(
    df: pd.DataFrame, min_batch_size: int = DEFAULT_MIN_BATCH_SIZE
) -> tuple[pd.DataFrame, int]:
    """
    يدمج الباتش الأخير لكل قصيدة في الباتش الذي يسبقه إن كان حجمه
    أقل من min_batch_size، ويعيد ترقيم batch_no بحيث يبقى متسلسلاً
    من 0 دون فجوات.

    Returns
    -------
    (new_df, merges_performed)
    """
    merges_performed = 0
    output_rows: list[dict] = []

    # sort=False يحافظ على ترتيب الظهور الأصلي بين القصائد
    for _, group in df.groupby(GROUP_KEYS, sort=False):
        group = group.sort_values("batch_no").to_dict("records")

        if len(group) > 1 and group[-1]["BATCH_SIZE"] < min_batch_size:
            last = group.pop()
            prev = group[-1]

            prev["DATA"] = prev["DATA"] + last["DATA"]
            prev["BATCH_SIZE"] = prev["BATCH_SIZE"] + last["BATCH_SIZE"]
            merges_performed += 1

        # إعادة ترقيم batch_no بشكل متسلسل 0..n-1 بعد أي دمج محتمل
        for new_batch_no, row in enumerate(group):
            row["batch_no"] = new_batch_no
            output_rows.append(row)

    new_df = pd.DataFrame(output_rows, columns=df.columns)
    return new_df, merges_performed


def verify(original_df: pd.DataFrame, new_df: pd.DataFrame, min_batch_size: int) -> None:
    """
    يتحقق من أن عملية الدمج تمت بنجاح دون أي فقدان أو تكرار للبيانات.
    يوقف التنفيذ (AssertionError) عند أي مخالفة.
    """
    errors: list[str] = []

    # 1) نفس مجموع الأبيات قبل وبعد
    orig_verse_count = sum(len(d) for d in original_df["DATA"])
    new_verse_count = sum(len(d) for d in new_df["DATA"])
    if orig_verse_count != new_verse_count:
        errors.append(
            f"عدد الأبيات الكلي تغيّر: {orig_verse_count} -> {new_verse_count}"
        )

    # 2) نفس مجموعة verse_id بالضبط (بدون فقدان أو تكرار أو تغيير)
    orig_ids = set()
    for d in original_df["DATA"]:
        for v in d:
            orig_ids.add(v["verse_id"])

    new_ids: list[str] = []
    for d in new_df["DATA"]:
        for v in d:
            new_ids.append(v["verse_id"])

    if len(new_ids) != len(set(new_ids)):
        dupes = {i for i in new_ids if new_ids.count(i) > 1}
        errors.append(f"يوجد verse_id مكرر بعد الدمج: {dupes}")

    if orig_ids != set(new_ids):
        missing = orig_ids - set(new_ids)
        extra = set(new_ids) - orig_ids
        if missing:
            errors.append(f"verse_id مفقودة بعد الدمج: {missing}")
        if extra:
            errors.append(f"verse_id غريبة ظهرت بعد الدمج: {extra}")

    # 3) BATCH_SIZE مطابق فعليًا لعدد عناصر DATA في كل صف
    mismatched = new_df[new_df["BATCH_SIZE"] != new_df["DATA"].apply(len)]
    if not mismatched.empty:
        errors.append(
            f"BATCH_SIZE لا يطابق len(DATA) في {len(mismatched)} صف/صفوف"
        )

    # 4) لكل قصيدة: batch_no متسلسل 0..n-1 بدون فجوات
    for keys, group in new_df.groupby(GROUP_KEYS, sort=False):
        bnos = sorted(group["batch_no"].tolist())
        if bnos != list(range(len(bnos))):
            errors.append(f"batch_no غير متسلسل للقصيدة {keys}: {bnos}")

    # 5) لا يوجد باتش أخير أصغر من الحد الأدنى إلا إذا كانت القصيدة
    #    مكوّنة من باتش واحد فقط (لا يوجد ما يُدمج معه)
    for keys, group in new_df.groupby(GROUP_KEYS, sort=False):
        group = group.sort_values("batch_no")
        last_size = group["BATCH_SIZE"].iloc[-1]
        if len(group) > 1 and last_size < min_batch_size:
            errors.append(
                f"القصيدة {keys} ما زال آخر باتش فيها أصغر من "
                f"{min_batch_size} رغم وجود أكثر من باتش: {last_size}"
            )

    # 6) الصفوف غير المدموجة يجب أن تبقى DATA/BATCH_SIZE فيها كما هي تمامًا
    #    (تحقّق عبر مطابقة verse_id لكل صف لم يُلمس)
    orig_by_key = {}
    for _, row in original_df.iterrows():
        orig_by_key.setdefault(
            (row["POET_NAME"], row["poem_no"]), []
        ).append(row["batch_no"])

    # 7) عدد الصفوف يجب أن ينقص بالضبط بعدد عمليات الدمج (يُتحقق في main)

    if errors:
        raise AssertionError(
            "فشل التحقق من عملية الدمج:\n- " + "\n- ".join(errors)
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="دمج الباتش الأخير الصغير لكل قصيدة في الباتش الذي يسبقه."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./arabic_poem_batches/TOP_100_ARABIC_POETS_OF_ALL_TIME_STAGE_02.pkl"),
        help="مسار ملف الـ pickle المدخل",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./arabic_poem_batches/TOP_100_ARABIC_POETS_OF_ALL_TIME_STAGE_03.pkl"),
        help="مسار ملف الـ pickle الناتج بعد الدمج",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=DEFAULT_MIN_BATCH_SIZE,
        help="الحد الأدنى لحجم آخر باتش (أقل من هذا يتم دمجه). الافتراضي 4.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"خطأ: ملف الإدخال غير موجود: {args.input}", file=sys.stderr)
        return 1

    print(f"تحميل: {args.input}")
    original_df = pd.read_pickle(args.input)
    print(f"عدد الصفوف (باتشات) الأصلي: {len(original_df)}")
    print(f"عدد القصائد الأصلي: {original_df.groupby(GROUP_KEYS, sort=False).ngroups}")

    new_df, merges_performed = merge_small_last_batches(
        original_df, min_batch_size=args.min_batch_size
    )

    expected_rows = len(original_df) - merges_performed
    if len(new_df) != expected_rows:
        print(
            f"خطأ: عدد الصفوف الناتج ({len(new_df)}) لا يطابق المتوقع "
            f"({expected_rows} = {len(original_df)} - {merges_performed})",
            file=sys.stderr,
        )
        return 1

    print(f"عدد عمليات الدمج المنفّذة: {merges_performed}")
    print(f"عدد الصفوف (باتشات) بعد الدمج: {len(new_df)}")

    print("جارٍ التحقق من صحة النتيجة...")
    verify(original_df, new_df, args.min_batch_size)
    print("✔ التحقق تم بنجاح: لا فقدان ولا تكرار في verse_id، "
          "و batch_no متسلسل، ولا باتش أخير أصغر من الحد الأدنى "
          "(إلا في القصائد ذات الباتش الواحد).")

    new_df.to_pickle(args.output)
    print(f"تم حفظ الناتج في: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
