# Midas - 投資情報収集・分析サーバー

## 0. 設計思想・投資哲学（Midas Core Principles）

### 0.1 基本思想

サーバント Midas は、世界・社会・技術の構造変化を継続的に観測し、長期的に価値が集中・移動する地点を見極めるための**投資意思決定支援エージェント群**である。

- 投資対象は「銘柄」ではなく「世の中の変化（構造変化）」である
- 価格は結果であり、判断材料の主軸には置かない
- 平常時は何もしない（Hold をデフォルトとする）
- 売却判断は価格ではなく Thesis（前提条件）の破綻によってのみ行う

### 0.2 想定する投資スタイル

- 長期投資（5〜10年スパン）
- 技術転換・制度転換・行動様式の変化に伴う価値移動への投資
- 王者（Dominant Platform）の優位条件崩壊と次の勝者の特定

---

## 1. システム概要

### 1.1 目的

投資に関する情報を自動収集・整理し、構造変化および前提条件の成立・破綻を継続的に監視することで、長期投資における意思決定を支援する。

### 1.2 対象ユーザー

- 長期資産形成を目的とする個人投資家

### 1.3 機能概要

| 機能 | 内容 |
|------|------|
| **買い調査** | 世の中の変化を調べ、恩恵を受ける企業を特定。企業固有のリスク要因（経営者、技術者流出、支配構造等）も確認 |
| **売り調査** | 保有銘柄を管理し、世の中の変化や同業他社の躍進など外部リスク要因を監視 |
| **未来洞察** | ニュースから未来シグナルを抽出し、投資テーマと受益企業を特定 |
| **企業発見** | 未来予測に基づき、実現に重要な役割を果たす企業を特定 |
| **学習** | 株価が急変（1ヶ月で3倍or1/3）した事例を振り返り、背景の構造変化を分析。得られた知見で調査ロジックを改善 |

### 1.4 対象投資商品

- 株式（米国、その他）

### 1.5 情報選別ポリシー

| 区分 | 内容 |
|------|------|
| **収集する** | 技術・制度・行動様式の変化に関わるもの |
| **収集しない** | 株価・相場などの市場ノイズ、決算・人事などの単発イベント、短期的な景気・センチメント情報 |

> **ポイント**: 政府情報は米国とそれ以外を区別する。日本を特別扱いはしない。

---

## 2. 機能要件

### 2.1 企業分析機能（Company Analysis）

- 企業固有のリスク要因を分析（経営者の性質、技術者の流出、支配構造、ガバナンス等）
- 財務情報を補助的に用い、前提条件（Thesis）の変化を検知
- 企業が属する構造やテーマとの整合性を評価

### 2.2 ポートフォリオ管理機能（Portfolio Management）

- 保有銘柄およびテーマ構成を管理
- 各投資先の前提条件（Thesis）の状態を管理
- 前提条件の破綻や構造変化を検知
- 売却検討が必要な状態を段階的に提示
- 売却は自動実行せず、人間の判断を前提とする

### 2.3 ニュース監視機能（News Watch）

世の中の構造や前提条件に関係するニュースのみを監視対象とする。

| 区分 | 監視対象 |
|------|----------|
| **米国政府ウォッチ** | 法案・規制・政策・国家戦略 |
| **その他政府ウォッチ** | 米国以外の政府・規制当局の制度変更や政策（日本を特別扱いしない） |
| **新技術ウォッチ** | 技術の実用化、標準化、ボトルネックの移動 |
| **その他ニュースウォッチ** | 上記に属さないが構造や前提条件に影響する情報 |

### 2.4 株式市場監視機能（Market Watch）

- 市場全体の異常や変調を監視
- 価格情報は異常検知や振り返り用途に限定
- 売買判断の直接材料としては使用しない

### 2.5 学習機能（Learning）

- 大きな価格変動（1ヶ月で3倍or1/3）が発生した事例を振り返る
- 背景となった世の中の変化を整理
- 得られた知見を企業分析・ニュース監視・売却判断の改善に反映

### 2.6 未来洞察機能（Future Insight）

- 収集したニュースから「未来シグナル」を抽出
  - 技術シフト、規制変更、行動変容、地政学、プラットフォームシフト等
- 複数のシグナルを統合して「投資テーマ」を合成
- 各テーマに対する受益者（企業・セクター・技術）を特定
- 時間軸（1-2年 / 3-5年 / 5-10年）と確信度を評価

### 2.7 重要企業発見機能（Critical Company Finder）

- 未来予測（例：「EV が主流になる」）を入力として受け取る
- 予測実現に必要な「重要コンポーネント」を分析
  - 技術、インフラ、サービス、素材、規制など
- 各コンポーネントで重要な役割を果たす企業を特定
  - 市場リーダー、挑戦者、ニッチプレイヤー、新興企業
- 競争優位性と確信度を評価
- 投資示唆とリスクを整理

---

## 3. 技術スタック

| カテゴリ | 技術 |
|----------|------|
| 言語 | Python 3.11+ |
| エージェント | LangGraph / LangChain |
| LLM | Google Gemini API |
| 株価取得 | yfinance, finvizfinance |
| HTTP | httpx |
| データモデル | Pydantic |
| データ保存 | JSON ファイル |

---

## 4. データソース

| 種類 | API |
|------|-----|
| 株価 | Yahoo Finance (yfinance) |
| 株式スクリーニング | FINVIZ (finvizfinance) |
| ニュース | RSS Feeds, Google News RSS |
| 政府情報 | 各政府機関 RSS（White House, Congress, SEC, EU, etc.） |

---

## 5. 開発フェーズ

### Phase 1: MVP
- [x] プロジェクト構造のセットアップ
- [x] 株価データ収集（yfinance）
- [x] 株式スクリーニング（FINVIZ）
- [x] 基本的な企業分析機能
- [x] CLI でのレポート出力

### Phase 2: ニュース監視
- [x] ニュース収集エージェント（米国政府）
- [x] ニュース収集エージェント（新技術）
- [x] ニュース収集エージェント（その他政府）
- [x] ニュース収集エージェント（一般ニュース）
- [x] 情報選別フィルタリング（LLM）
- [x] ポートフォリオ管理機能
- [x] 未来洞察エージェント
- [x] 重要企業発見エージェント

### Phase 3: 学習・改善
- [x] 株価急変事例の収集・分析
- [x] 知見のフィードバックループ
- [ ] 通知機能（Slack / LINE）
- [ ] PostgreSQL でのデータ永続化

---

## 6. エージェント

### 6.1 エージェント一覧

#### foresight_manager（未来予測管理エージェント）

未来予測の管理を担うオーケストレーターエージェント。4種の Watcher エージェント（tech_news_watcher, us_gov_watcher, other_gov_watcher, general_news_watcher）と prediction_monitor からの情報を統合し、foresight_to_company_translator に渡す。
各エージェントに対して，前回どういう情報が足りなかったから、今回はこういう情報を追加的に集めてほしいなどの指示を出す。
情報収集後，foresightについてまとめたスライドをGoogleスライドに出力する。

- 入力元: ここがSTART地点. 下記の出力先からのフィードバックを受領
- 出力先: 4種の Watcher エージェントと prediction_monitor，foresight_to_company_translator，Googleスライド

---

#### tech_news_watcher（テクノロジーニュース監視エージェント）

テクノロジー関連の情報を収集するエージェント。
model_calibration_agentからの指示を受けて情報収集先を見直す機能を備える。

- **使用ツール**: RSS Fetcher, LLM
- **データソース**: Ars Technica, TechCrunch, Wired, MIT Tech Review 等
- **入出力**: foresight_manager から起動され、収集結果を foresight_manager に返す

---

#### us_gov_watcher（米国政府監視エージェント）

米国政府の公式情報を収集するエージェント。法案・規制・政策・国家戦略など、構造変化に関わる政府動向を監視する。
model_calibration_agentからの指示を受けて情報収集先を見直す機能を備える。

- **使用ツール**: RSS Fetcher, LLM
- **データソース**: White House, Congress, Federal Register, SEC, USTR
- **入出力**: foresight_manager から起動され、収集結果を foresight_manager に返す

---

#### other_gov_watcher（他国政府監視エージェント）

米国以外の政府・国際機関の情報を収集するエージェント。日本を特別扱いせず、グローバルな視点で規制・政策変更を監視する。
model_calibration_agentからの指示を受けて情報収集先を見直す機能を備える。

- **使用ツール**: RSS Fetcher, LLM
- **データソース**: EU, UK, 中国, 日本, IMF, World Bank 等
- **入出力**: foresight_manager から起動され、収集結果を foresight_manager に返す

---

#### general_news_watcher（一般ニュース監視エージェント）

経済・ビジネス関連の一般ニュースを収集するエージェント。上記3つの Watcher でカバーしきれない構造変化情報を拾う。
model_calibration_agentからの指示を受けて情報収集先を見直す機能を備える。

- **使用ツール**: RSS Fetcher, LLM
- **データソース**: Yahoo Finance, Bloomberg, Reuters, CNBC 等
- **入出力**: foresight_manager から起動され、収集結果を foresight_manager に返す

---

#### prediction_monitor（予測監視エージェント）

社会変化を分析するエージェント。McKinsey、BCG、WEF などの年次展望記事をスキャンし、中長期の社会変化トレンドを検出する。
model_calibration_agentからの指示を受けて情報収集先を見直す機能を備える。

- **使用ツール**: Google News RSS, LLM
- **入出力**: foresight_manager から起動され、分析結果を foresight_manager に返す

---

#### foresight_to_company_translator（展望-企業変換エージェント）

未来予測から重要企業を特定するエージェント。3段階の LLM 分析（バリューチェーン分解 → 企業特定 → ボトルネック分析）により、投資候補となる企業を抽出する。

- **使用ツール**: LLM（3段階分析）
- **入力元**: prediction_monitor，company_watcher
- **出力先**: company_watcher, portfolio_manager

---

#### company_watcher（企業監視エージェント）

今持っていないその銘柄を買うべきかどうか、または今持っている銘柄をホールドし続けてよいかを判断し，portfolio_managerに通知する

今持っていない銘柄の場合
・事業内容のうち，未来予測と関連する事業の割合
・競争優位性
・経営者のカリスマ性，能力
・PERなど，今の株価が高すぎないか

今持っている銘柄の場合
・ライバル企業における著しい競争優位の発生
・経営者の退任
・致命的な訴訟
・規制当局による認可拒否
・PERなど，今の株価が安すぎないか

- **使用ツール**: yfinance, Company News Fetcher, LLM
- **入力**: foresight_to_company_translator, portfolio_manager, CLI
- **出力**: portfolio_manager

---

#### price_event_analyzer（株価イベント分析エージェント）

過去1ヶ月で株価が3倍または3分の1になった銘柄を洗い出す
その銘柄の過去2ヶ月のニュースを検索し，結果をmodel_calibration_agentに渡す

- **使用ツール**: yfinance, Company News Fetcher, Gemini LLM
- **入力**: portfolio_manager, CLI
- **出力**: model_calibration_agent

---

#### portfolio_manager（ポートフォリオ管理エージェント）

保有銘柄のポートフォリオを管理し，あらたな銘柄の買付提案，保有銘柄の保有継続提案を行う
foresight破綻の兆候を検出する
現状と提案内容をGoogleスライドに出力する

- **使用ツール**: yfinance, Gemini LLM
- **入力**: portfolio.json, price_event_analyzer, foresight_to_company_translator
- **出力**: Googleスライド 

---

#### model_calibration_agent（モデル校正エージェント）

株価急変事例から学習するエージェント。
price_event_analyzerから通知を受けた銘柄と過去のニュースを分析し，次のアクションを行う

1ヶ月で3倍上昇の場合
・未来予測に関係する動きなのか，個別企業の動きなのかを判別
・未来予測に関係する動きだった場合，foresight_managerからデータを取得し，その未来予測がカバーできているか確認する
・カバーできていない場合，ニュースウォッチャーにそれをカバーするよう指示する

1ヶ月で1/3下落の場合
・company_watcherからデータを取得し，その株価の下落の要因の収集ができているか確認する
・カバーできていない場合，company_watcherにそれをカバーするよう指示する

- **使用ツール**: Stock Screener, yfinance, Company News Fetcher, Gemini LLM
- **入力**: price_event_analyzer, Stock Screener
- **出力**: 全エージェント（知見フィードバック）


### 6.2 CLI コマンドとエージェントの対応

```bash
# データ収集
midas collect --source all          # 全 News Watcher 実行
midas collect --source us-gov       # us_gov_watcher
midas collect --source tech         # tech_news_watcher

# 未来洞察
midas insight --days 7              # forsight_manager

# 年次展望
midas prediction-monitor scan --year 2026  # prediction_monitor

# 企業発見
midas find-companies "EVが主流になる"  # foresight_to_company_translator

# 企業分析
midas analyze AAPL --mode risk      # company_watcher
midas analyze AAPL --mode price     # price_event_analyzer
midas analyze AAPL --mode full      # 両方実行

# ポートフォリオ
midas portfolio analyze             # portfolio_manager

# 学習
midas learn scan --period month     # model_calibration_agent（急変事例スキャン）
midas learn insights                # 蓄積した知見の表示
midas learn cases                   # 分析済み事例の表示
```
