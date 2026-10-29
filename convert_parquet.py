import pandas as pd

def main():
    df = pd.read_parquet("train-00000-of-00001.parquet")

    print("전체 데이터 수:", len(df))
    print(df.head())

    df.to_csv(
        "kor_3i4k.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("CSV 변환 완료!")


if __name__ == "__main__":
    main()