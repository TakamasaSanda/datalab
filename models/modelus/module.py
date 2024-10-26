# left Outer Joinした後に残る重複しているカラムを消すためのヘルパー関数
def left_merge(data1, data2, on):
    # data1とdata2_tempをonのキーを用いて、Left Outer Joinする
    result = data1.merge(data2, on=on, how="left")
    # data2のカラム名のうち、onに存在していないカラム名の一覧を取得する
    data2_columns = [f for f in data2.columns if f not in on]
    # data2のカラムに存在しているon以外のカラム名の値だけを取り出す
    result = result[data2_columns]
    return result

# day1からday2までの日数を求める関数
def diff_of_days(day1, day2):
    days = (parse(day1[:10]) - parse(day2[:10])).days
    return days

# 日付を加算する関数
def date_add_days(start_date, days):
    end_date = parse(start_date[:10]) + timedelta(days=days)
    end_date = end_date.strftime("%Y-%m-%d")
    return end_date

def calculate_label(end_date, n_day, df_visit, df_date):
    # データの期間を絞り込む範囲を指定するためにlabelの終了日を算出
    label_end_date = date_add_days(end_date, n_day)
    # データの期間を絞り込む。visit_dateがlabel_end_dateより前で、visit_dateがend_dateより後である期間のデータだけを取得する
    label = df_visit[
        (df_visit["visit_date"] < label_end_date) & (df_visit["visit_date"] >= end_date)
    ].copy()
    # end_dateをカラムとして追加する
    label["end_date"] = end_date
    # visit_dateからend_dateまでの日数を求める
    label["diff_of_day"] = label["visit_date"].apply(
        lambda x: diff_of_days(x, end_date)
    )
    # visit_dateの月の情報を抽出する(例: 2022/01/02 -> 1)
    label["month"] = label["visit_date"].str[5:7].astype(int)
    # visit_dateの年の情報を抽出する(例: 2022/01/01 -> 2022)
    label["year"] = label["visit_date"].str[:4].astype(int)
    # 3日後・2日後・1日後・1日前が休日かどうかを算出する
    for i in [3, 2, 1, -1]:
        date_info_temp = df_date.copy()
        # visit_dateカラムの値にi日加算した日付でvisit_dateカラムを上書きする
        date_info_temp["visit_date"] = date_info_temp["visit_date"].apply(
            lambda x: date_add_days(x, i)
        )
        # date_info_tempのholiday_flgカラムとholiday_flg2カラムをiの値を接尾語として使ってカラム名を変える
        # 次の行でのマージ時に同じカラム名のものをマージするとカラム名が機械的にholiday_flgならholiday_flg_xとholiday_flg_yに変更されてしまうためこうしている
        date_info_temp.rename(
            columns={
                "holiday_flg": "ahead_holiday_{}".format(i),
                "holiday_flg2": "ahead_holiday2_{}".format(i),
            },
            inplace=True,
        )
        # visit_dateカラムをキーにして、labelとdate_info_tempでLeft Outer Joinでマージする
        label = label.merge(date_info_temp, on=["visit_date"], how="left")
    label = label.reset_index(drop=True)
    return label

def calculate_store_visitor_feature(label, end_date, n_day, df_visit):
    # データの期間を絞り込む範囲を指定するために開始日を算出
    start_date = date_add_days(end_date, -n_day)
    # データの期間を絞り込む。visit_dateが開始日より後から終了日より前である期間のデータだけを取得する
    data_temp = df_visit[
        (df_visit.visit_date < end_date) & (df_visit.visit_date > start_date)
    ].copy()
    # 店舗IDごとに訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量で集計する
    result = data_temp.groupby(["air_store_id"], as_index=False)["visitors"].agg(
        {
            "store_min{}".format(n_day): "min",
            "store_mean{}".format(n_day): "mean",
            "store_median{}".format(n_day): "median",
            "store_max{}".format(n_day): "max",
            "store_count{}".format(n_day): "count",
            "store_std{}".format(n_day): "std",
            "store_skew{}".format(n_day): "skew",
        }
    )
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    result = left_merge(label, result, on=["air_store_id"]).fillna(0)
    return result


def calculate_store_week_feature(label, end_date, n_day, df_visit):
    # 開始日を算出
    start_date = date_add_days(end_date, -n_day)
    # データの期間を絞り込む。visit_dateが開始日より後から終了日より前である期間のデータだけを取得する
    data_temp = df_visit[
        (df_visit.visit_date < end_date) & (df_visit.visit_date > start_date)
    ].copy()
    # 曜日ごとの訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量で集計する
    result = data_temp.groupby(["air_store_id", "dow"], as_index=False)["visitors"].agg(
        {
            "store_dow_min{}".format(n_day): "min",
            "store_dow_mean{}".format(n_day): "mean",
            "store_dow_median{}".format(n_day): "median",
            "store_dow_max{}".format(n_day): "max",
            "store_dow_count{}".format(n_day): "count",
            "store_dow_std{}".format(n_day): "std",
            "store_dow_skew{}".format(n_day): "skew",
        }
    )
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    result = left_merge(label, result, on=["air_store_id", "dow"]).fillna(0)
    return result


def calculate_store_week_diff_feature(label, end_date, n_day, df_visit):
    # 開始日を算出
    start_date = date_add_days(end_date, -n_day)
    # データの期間を絞り込む。visit_dateが開始日より後から終了日より前である期間のデータだけを取得する
    data_temp = df_visit[
        (df_visit.visit_date < end_date) & (df_visit.visit_date > start_date)
    ].copy()
    # 日ごとの訪問者数を行方向から列方向のデータに変更する
    result = data_temp.set_index(["air_store_id", "visit_date"])["visitors"].unstack()
    # 訪問者数の1日前との差分を求める
    result = result.diff(axis=1).iloc[:, 1:]
    # カラム名を取得
    column_names = result.columns
    # 差分の平均、標準偏差、最大、最小の統計量を出して特徴量にする
    result["store_diff_mean"] = np.abs(result[column_names]).mean(axis=1)
    result["store_diff_std"] = result[column_names].std(axis=1)
    result["store_diff_max"] = result[column_names].max(axis=1)
    result["store_diff_min"] = result[column_names].min(axis=1)
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    result = left_merge(
        label,
        result[
            ["store_diff_mean", "store_diff_std", "store_diff_max", "store_diff_min"]
        ],
        on=["air_store_id"],
    ).fillna(0)
    return result


def calculate_store_all_week_feature(label, end_date, n_day, df_visit):
    # 開始日を算出
    start_date = date_add_days(end_date, -n_day)
    # データの期間を絞り込む。visit_dateが開始日より後から終了日より前である期間のデータだけを取得する
    data_temp = df_visit[
        (df_visit.visit_date < end_date) & (df_visit.visit_date > start_date)
    ].copy()
    # 店舗・曜日ごとの訪問者数を平均、中央値、最大値、カウントの4つの統計量を出す
    result_temp = data_temp.groupby(["air_store_id", "dow"], as_index=False)[
        "visitors"
    ].agg(
        {
            "store_dow_mean{}".format(n_day): "mean",
            "store_dow_median{}".format(n_day): "median",
            "store_dow_sum{}".format(n_day): "max",
            "store_dow_count{}".format(n_day): "count",
        }
    )
    result = pd.DataFrame()
    # 全ての曜日に対して、特徴量を生成する
    for i in range(7):
        # 曜日の番号で絞り込みをかける。カラム名の前に曜日の番号を付与する
        result_sub = (
            result_temp[result_temp["dow"] == i]
            .copy()
            .set_index("air_store_id")
            .add_prefix(str(i))
        )
        # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
        result_sub = left_merge(label, result_sub, on=["air_store_id"]).fillna(0)
        # 作成した曜日の特徴量を列に追加
        result = pd.concat([result, result_sub], axis=1)
    return result


def calculate_store_holiday_feature(label, end_date, n_day, df_visit):
    # 開始日を算出
    start_date = date_add_days(end_date, -n_day)
    # データの期間を絞り込む。visit_dateが開始日より後から終了日より前である期間のデータだけを取得する
    data_temp = df_visit[
        (df_visit.visit_date < end_date) & (df_visit.visit_date > start_date)
    ].copy()
    # date_info.csvのholiday_flgを使って休日と平日でそれぞれ訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量を集計する
    result1 = data_temp.groupby(["air_store_id", "holiday_flg"], as_index=False)[
        "visitors"
    ].agg(
        {
            "store_holiday_min{}".format(n_day): "min",
            "store_holiday_mean{}".format(n_day): "mean",
            "store_holiday_median{}".format(n_day): "median",
            "store_holiday_max{}".format(n_day): "max",
            "store_holiday_count{}".format(n_day): "count",
            "store_holiday_std{}".format(n_day): "std",
            "store_holiday_skew{}".format(n_day): "skew",
        }
    )
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    result1 = left_merge(label, result1, on=["air_store_id", "holiday_flg"]).fillna(0)

    # カレンダーの日付が土曜日、日曜日の時もしくは店舗の休日の時とそうでない時でそれぞれ訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量を集計する
    result2 = data_temp.groupby(["air_store_id", "holiday_flg2"], as_index=False)[
        "visitors"
    ].agg(
        {
            "store_holiday2_min{}".format(n_day): "min",
            "store_holiday2_mean{}".format(n_day): "mean",
            "store_holiday2_median{}".format(n_day): "median",
            "store_holiday2_max{}".format(n_day): "max",
            "store_holiday2_count{}".format(n_day): "count",
            "store_holiday2_std{}".format(n_day): "std",
            "store_holiday2_skew{}".format(n_day): "skew",
        }
    )
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    result2 = left_merge(label, result2, on=["air_store_id", "holiday_flg2"]).fillna(0)
    # ここで作成した特徴量を結合する
    result = pd.concat([result1, result2], axis=1)
    return result


def calculate_first_last_time_feature(label, end_date, n_day, df_visit):
    # 開始日を算出
    start_date = date_add_days(end_date, -n_day)
    # データの期間を絞り込む。visit_dateが開始日より後から終了日より前である期間のデータだけを取得する
    data_temp = df_visit[
        (df_visit.visit_date < end_date) & (df_visit.visit_date > start_date)
    ].copy()
    # 訪問日付を昇順でソートする
    data_temp = data_temp.sort_values("visit_date")

    # 絞り込んだ期間のデータの最大と最小の日付の終了日からの日数を集計する
    result = (
        data_temp.groupby("air_store_id")["visit_date"]
        .agg(
            [
                lambda x: diff_of_days(end_date, np.min(x)),
                lambda x: diff_of_days(end_date, np.max(x)),
            ]
        )
        .rename(columns={"<lambda_0>": "first_time", "<lambda_1>": "last_time"})
    )
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    result = left_merge(label, result, on=["air_store_id"]).fillna(0)
    return result


def calculate_reserve_feature(label, end_date, n_day, df_reserve):
    # labelの終了日を算出する
    label_end_date = date_add_days(end_date, n_day)
    # visit_dateが終了日以降からlabelの終了日より前までの期間であり、予約日が終了日より前のデータだけに絞り込む。
    air_reserve_temp = df_reserve[
        (df_reserve.visit_date >= end_date)
        & (df_reserve.visit_date < label_end_date)
        & (df_reserve.reserve_date < end_date)
    ].copy()
    # Left Outer Joinで絞り込んだデータとdf_reserveを結合する
    air_reserve_temp = air_reserve_temp.merge(df_reserve, on="air_store_id", how="left")
    # 訪問日時から予約日時までの日数をカラムに加える
    air_reserve_temp["diff_time"] = (
        pd.to_datetime(df_reserve["visit_datetime"])
        - pd.to_datetime(df_reserve["reserve_datetime"])
    ).dt.days
    air_reserve_temp = air_reserve_temp.merge(df_reserve, on="air_store_id")
    # 店舗・訪問日ごとに予約した訪問者数の合計とカウントを集計する
    air_result = air_reserve_temp.groupby(["air_store_id", "visit_date"])[
        "reserve_visitors"
    ].agg(
        air_reserve_visitors="sum",
        air_reserve_count="count",
    )
    # Nanを0で埋めて整形する
    air_result = air_result.unstack().fillna(0).stack()
    # 店舗・訪問日ごとに訪問日時から予約日時までの日数の平均を求める。
    air_store_diff_time_mean = air_reserve_temp.groupby(["air_store_id", "visit_date"])[
        "diff_time"
    ].agg(air_store_diff_time_mean="mean")

    # 店舗の区別をせず、訪問日ごとに訪問日時から予約日時までの日数の平均を求める。
    air_diff_time_mean = air_reserve_temp.groupby(["visit_date"])["diff_time"].agg(
        air_diff_time_mean="mean"
    )
    # 店舗の区別をせず、訪問日ごとに予約者数の合計、カウントを求める。
    air_date_result = air_reserve_temp.groupby(["visit_date"])["reserve_visitors"].agg(
        air_date_visitors="sum", air_date_count="count"
    )
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    air_result = left_merge(
        label, air_result, on=["air_store_id", "visit_date"]
    ).fillna(0)
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    air_store_diff_time_mean = left_merge(
        label, air_store_diff_time_mean, on=["air_store_id", "visit_date"]
    ).fillna(0)
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    air_date_result = left_merge(label, air_date_result, on=["visit_date"]).fillna(0)
    # labelにconcatで連結できるようにするために、labelとここで作成した特徴量を結合して、特徴量のカラムだけを取り出す。
    air_diff_time_mean = left_merge(
        label, air_diff_time_mean, on=["visit_date"]
    ).fillna(0)
    # ここで作成した特徴量を結合する
    result = pd.concat(
        [air_result, air_date_result, air_store_diff_time_mean, air_diff_time_mean],
        axis=1,
    )
    return result


def make_features(end_date, n_day, df_visit, df_reserve, df_date):
    t0 = time.time()
    result = []

    # n_day-1日間のデータから終了日、終了日からの日数、月、年、3日後・2日後・1日後・1日前が休日かどうかの特徴量に追加
    label = calculate_label(end_date, n_day, df_visit, df_date)
    result.append(label)

    # 56-2=54日間・28-2=26日間・14-2=12日間の店舗IDごとの訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量の特徴量に追加
    result.append(calculate_store_visitor_feature(label, end_date, 56, df_visit))
    result.append(calculate_store_visitor_feature(label, end_date, 28, df_visit))
    result.append(calculate_store_visitor_feature(label, end_date, 14, df_visit))

    # 56-2=54日間・28-2=26日間・14-2=12日間の曜日ごとの訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量の特徴量に追加
    result.append(calculate_store_week_feature(label, end_date, 56, df_visit))
    result.append(calculate_store_week_feature(label, end_date, 28, df_visit))
    result.append(calculate_store_week_feature(label, end_date, 14, df_visit))

    # 58日-2日=56日(4週間)の訪問者数の1日前との差分の平均、標準偏差、最大、最小の統計量の特徴量に追加
    result.append(calculate_store_week_diff_feature(label, end_date, 58, df_visit))

    # 1000-2=998日間のデータから店舗・曜日ごとの訪問者数を平均、中央値、最大値、カウントの統計量から曜日による特徴量に追加
    result.append(calculate_store_all_week_feature(label, end_date, 1000, df_visit))

    # 1000-2=998日間のデータから休日と平日でそれぞれ訪問者数を最小値、平均、中央値、最大値、カウント、標準偏差、歪度の各統計量の特徴量に追加
    result.append(calculate_store_holiday_feature(label, end_date, 1000, df_visit))

    # 予約情報を用いて以下の特徴量を作成する。
    # ・店舗・訪問日ごとに予約した訪問者数の合計とカウント
    # ・店舗・訪問日ごとに訪問日時から予約日時までの日数の平均
    # ・店舗の区別をせず、訪問日ごとに訪問日時から予約日時までの日数の平均
    # ・店舗の区別をせず、訪問日ごとに予約者数の合計とカウント
    result.append(calculate_reserve_feature(label, end_date, n_day, df_reserve))

    # 1000-2=998日間のデータからのうち、最大の日付と最小の日付からの終了日からの日数を特徴量に追加
    result.append(calculate_first_last_time_feature(label, end_date, 1000, df_visit))

    print("merge...")
    # 全特徴量をDataFrameの形で結合してまとめる
    result = pd.concat(result, axis=1)

    print("data shape：{}".format(result.shape))
    print("spending {}s".format(time.time() - t0))
    return result


def feature_engineering(df_visit, df_reserve, df_date, is_log_transform=False):
    df_visit_temp = df_visit.copy()
    if is_log_transform:
        # 訪問者数に対数をとる
        df_visit_temp["visitors"] = np.log1p(df_visit_temp["visitors"])
    train_feat = pd.DataFrame()
    # 開始日を2017年2月13日に設定する
    start_date = "2017-02-13"
    # 開始日から58週間前まで1週ずつ遡って特徴量を作成する
    for i in range(58):
        train_feat_sub = make_features(
            date_add_days(start_date, i * (-7)), 39, df_visit_temp, df_reserve, df_date
        )
        train_feat = pd.concat([train_feat, train_feat_sub])
    # 開始日の1週間後から5週間後までの特徴量を作成する
    for i in range(1, 6):
        train_feat_sub = make_features(
            date_add_days(start_date, i * (7)),
            42 - (i * 7),
            df_visit_temp,
            df_reserve,
            df_date,
        )
        train_feat = pd.concat([train_feat, train_feat_sub])
    # テスト用の特徴量を作成する
    test_feat = make_features(
        date_add_days(start_date, 42), 39, df_visit_temp, df_reserve, df_date
    )

    # 予測対象のカラム名のリストを作る
    predictors = [
        f
        for f in test_feat.columns
        if f
        not in (
            [
                "id",
                "air_store_id",
                "visit_date",
                "end_date",
                "air_area_name",
                "visitors",
                "month",
                "air_genre_name",
            ]
        )
    ]
    return train_feat, test_feat, predictors, train_feat["visitors"]

"""
# ----------------
air_visit_max, air_reserve, date_info

air_visit_max-> joined_n_store_df
air_reserve-> air_reserve
 date_info-> date_info

# ----------------
"""

# 予測対象の訪問数に対数をとった場合の特徴量を作成する
train_log, test_log, predictors_log, target_log = feature_engineering(
    air_visit_max, air_reserve, date_info, is_log_transform=True
)
# 予測対象の訪問数から特徴量を作成する
train, test, predictors, target = feature_engineering(
    air_visit_max, air_reserve, date_info, is_log_transform=False
)

# LightGBMの学習する際のパラメーター
params = {
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "sub_feature": 0.7,
    "num_leaves": 60,
    "min_data": 1,
    "min_hessian": 1,
    "verbose_eval": -1,
}

# 訪問者数に対数をとった場合のデータセットを学習に使う特徴量と目的変数をLightGBM用のDatasetに格納する
lgb_train_log = lgb.Dataset(train_log[predictors_log], target_log)
# LightGBMを学習する
gbm_log = lgb.train(params, lgb_train_log, 1000)
# LightGBMのモデルを使って、テスト用の特徴量から訪問者数を予測
pred_log = gbm_log.predict(test_log[predictors_log])

# 訪問者数のデータセットを学習に使う特徴量と目的変数をLightGBM用のDatasetに格納する
lgb_train = lgb.Dataset(train[predictors], target)
# LightGBMを学習する
gbm = lgb.train(params, lgb_train, 1000)
# LightGBMのモデルを使って、テスト用の特徴量から訪問者数を予測
pred = gbm.predict(test[predictors])

# 教師データである訪問者数に対数をとった場合のデータセットを使って学習したモデルの訪問者数の予測精度を測るためにMean Absolute Errorを求める
mae_log = f"{mean_absolute_error(test['visitors'].values,np.expm1(pred_log)):.3f}"
# 訪問者数のデータセットを使って学習したモデルの訪問者数の予測精度を測るためにMean Absolute Errorを求める
mae = f"{mean_absolute_error(test['visitors'].values,pred):.3f}"

print(pd.DataFrame([[mae_log, mae]], columns=["モデル1", "モデル2"], index=["mae"]))