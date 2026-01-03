# Midas コードベースマップ

## 1. プロジェクト概要

**Midas** は、長期投資家向けの投資意思決定支援エージェントシステム。世界・社会・技術の構造変化を継続的に監視して、価値が移動する地点を見極める。

### 設計思想

- 投資対象は「銘柄」ではなく「世の中の変化（構造変化）」
- 価格は結果であり、判断材料の主軸には置かない
- 平常時は Hold をデフォルト
- 売却判断は Thesis（前提条件）の破綻によってのみ行う

---

## 2. ファイル構成（Tree View）

```
Midas/
├── .claude/                         # Claude Code 設定
│   ├── commands/
│   │   └── analyze-project.md       # /analyze-project カスタムコマンド
│   └── settings.local.json          # 権限設定
├── data/                            # データ保存ディレクトリ
│   ├── company_analysis/            # 企業分析結果
│   ├── future_insights/             # 未来洞察レポート（prediction_monitor/と同義）
│   ├── general/                     # 一般ニュース
│   ├── logs/                        # 実行ログ
│   ├── news/                        # ニュースキャッシュ
│   ├── other_gov/                   # 米国以外政府ニュース
│   ├── portfolio/                   # ポートフォリオデータ
│   ├── prediction_analysis/         # 予測分析結果
│   ├── tech/                        # 技術ニュース
│   └── us_gov/                      # 米国政府ニュース
├── docs/
│   ├── requirements.md              # 要件定義書
│   └── CODEBASE_MAP.md              # 本ファイル
├── src/midas/
│   ├── __init__.py                  # パッケージ初期化
│   ├── __main__.py                  # python -m midas エントリーポイント
│   ├── main.py                      # CLI メインエントリーポイント
│   ├── config.py                    # 設定・環境変数管理
│   ├── models.py                    # Pydantic データモデル
│   ├── agents/                      # LangGraph エージェント群
│   │   ├── __init__.py
│   │   ├── us_gov_watcher.py        # 米国政府情報監視
│   │   ├── tech_news_watcher.py     # 技術ニュース監視
│   │   ├── other_gov_watcher.py     # 米国以外政府情報監視
│   │   ├── general_news_watcher.py  # 一般ニュース監視
│   │   ├── price_event_analyzer.py  # 株価イベント分析
│   │   ├── company_watcher.py       # 企業分析エージェント
│   │   ├── portfolio_manager.py     # ポートフォリオ管理
│   │   ├── prediction_monitor.py    # 年次社会変化分析
│   │   ├── model_calibration_agent.py # 株価急変からの学習
│   │   └── foresight_to_company_translator.py # 重要企業発見
│   └── tools/                       # ツール群
│       ├── __init__.py
│       ├── rss_fetcher.py           # RSS フィード取得
│       ├── stock_screener.py        # 株式スクリーニング
│       ├── portfolio_manager.py     # ポートフォリオ管理
│       └── company_news_fetcher.py  # 企業ニュース取得
├── .env                             # 環境変数（Git 管理外）
├── .gitignore
├── pyproject.toml                   # プロジェクト設定
└── uv.lock                          # 依存関係ロック
```

---

## 3. Claude Code 設定 (.claude/)

### settings.local.json

```json
{
  "permissions": {
    "allow": [
      "mcp__filesystem__list_directory",
      "mcp__git__git_status",
      "mcp__git__git_log",
      "mcp__filesystem__directory_tree"
    ]
  }
}
```

### commands/analyze-project.md

`/analyze-project` カスタムコマンドの定義。このコマンドを実行すると、プロジェクト構造をスキャンして本ファイルを更新する。

---

## 4. 各ファイルの詳細解説

### src/midas/main.py

**役割**: CLI エントリーポイント。全てのコマンドを処理

**データフロー**:
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   CLI       │ ───► │  argparse   │ ───► │  サブコマンド │
│ $ midas ... │      │  パース     │      │  実行        │
└─────────────┘      └─────────────┘      └─────────────┘
                                                 │
                     ┌───────────────────────────┼───────────────────────────┐
                     ▼                           ▼                           ▼
              ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
              │   collect   │            │   analyze   │            │  portfolio  │
              │  ニュース収集 │            │  企業分析   │            │ ポートフォリオ│
              └─────────────┘            └─────────────┘            └─────────────┘
```

**CLI コマンド一覧**:

| コマンド | 説明 | 例 |
|----------|------|-----|
| `collect` | ニュース収集 | `midas collect --source=us-gov` |
| `screen` | 株式スクリーニング | `midas screen --timeframe=week` |
| `analyze` | 企業分析 | `midas analyze AAPL --mode=full` |
| `portfolio show` | ポートフォリオ表示 | `midas portfolio show` |
| `portfolio import` | CSV インポート | `midas portfolio import --file=holdings.csv` |
| `portfolio add` | 手動追加 | `midas portfolio add AAPL 10 150` |
| `portfolio update` | 価格更新 | `midas portfolio update` |
| `portfolio analyze` | LLM 分析 | `midas portfolio analyze` |
| `insight` | 未来洞察生成 | `midas insight --days=7` |
| `find-companies` | 重要企業発見 | `midas find-companies "EV will dominate"` |

---

### src/midas/config.py

**役割**: アプリケーション設定の一元管理

**定義変数**:

| 変数 | 型 | 説明 |
|------|-----|------|
| `GEMINI_API_KEY` | `str \| None` | Gemini API キー |
| `PROJECT_ROOT` | `Path` | プロジェクトルート |
| `DATA_DIR` | `Path` | データ保存ディレクトリ |
| `NEWS_DIR` | `Path` | ニュース保存ディレクトリ |
| `LLM_MODEL` | `str` | 使用モデル（`gemini-3-flash-preview`） |
| `LLM_MAX_TOKENS` | `int` | 最大トークン数（4096） |

---

### src/midas/models.py

**役割**: 全データモデルを Pydantic で定義

#### ニュース関連モデル

| モデル | 説明 |
|--------|------|
| `NewsCategory` | ニュースカテゴリ Enum（legislation, regulation, policy, etc.） |
| `NewsItem` | 単一ニュース記事 |
| `NewsCollection` | ニュースコレクション |

#### 企業分析モデル

| モデル | 説明 |
|--------|------|
| `StockMovement` | 株価変動 |
| `PriceEvent` | 重要な株価イベント |
| `CompanyNews` | 企業関連ニュース |
| `PriceEventAnalysis` | 株価イベント分析結果 |
| `NegativeInfo` | ネガティブ情報 |
| `CompanyAnalysis` | 企業総合分析 |

#### ポートフォリオ管理モデル

| モデル | 説明 |
|--------|------|
| `AccountType` | 口座種別 Enum（一般、特定、NISA 等） |
| `StockHolding` | 保有銘柄 |
| `Transaction` | 取引記録 |
| `Portfolio` | ポートフォリオ全体 |

#### 未来洞察モデル

| モデル | 説明 |
|--------|------|
| `SignalCategory` | シグナルカテゴリ Enum |
| `TimeHorizon` | 時間軸 Enum（near, medium, long） |
| `FutureSignal` | 未来シグナル |
| `Beneficiary` | 受益者 |
| `InvestmentTheme` | 投資テーマ |
| `FutureInsightReport` | 未来洞察レポート |

#### 予測分析モデル

| モデル | 説明 |
|--------|------|
| `CriticalComponent` | 重要コンポーネント |
| `CriticalCompany` | 重要企業 |
| `FuturePredictionAnalysis` | 予測分析結果 |

---

## 5. エージェント/ワークフロー

### 共通アーキテクチャ

全エージェントは LangGraph の `StateGraph` を使用した同様の構造:

```
┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────┐
│  fetch  │ ───► │ filter/ │ ───► │  save   │ ───► │ END │
│ (取得)  │      │ analyze │      │ (保存)  │      │     │
└─────────┘      └─────────┘      └─────────┘      └─────┘
```

### ニュース監視エージェント（4種）

| エージェント | ソース | 保存先 |
|-------------|--------|--------|
| `us_gov_watcher` | White House, Congress, Federal Register, SEC, USTR | `data/us_gov/` |
| `tech_news_watcher` | Ars Technica, TechCrunch, Verge, Wired, MIT Tech Review, etc. | `data/tech/` |
| `other_gov_watcher` | EU, UK, China, Japan, IMF, World Bank | `data/other_gov/` |
| `general_news_watcher` | Yahoo Finance, MarketWatch, Bloomberg, Reuters, etc. | `data/general/` |

**ワークフロー**:
```
┌─────────────┐      ┌──────────────┐      ┌───────────┐      ┌─────┐
│ fetch_news  │ ───► │ filter_news  │ ───► │   save    │ ───► │ END │
│ (RSS取得)   │      │ (LLMで判定)   │      │ (JSON保存)│      │     │
└─────────────┘      └──────────────┘      └───────────┘      └─────┘
```

**AgentState**:
```python
class AgentState(TypedDict):
    raw_items: list[NewsItem]
    filtered_items: list[NewsItem]
    saved_path: str | None
    error: str | None
```

### price_event_analyzer

**役割**: 株価の大きな変動（±5%以上）を検出し、原因を分析

**ワークフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────┐      ┌─────┐
│ fetch_prices │ ───► │ fetch_news   │ ───► │   analyze   │ ───► │ save │ ───► │ END │
│ (yfinance)   │      │ (Google News)│      │ (LLM分析)   │      │      │      │     │
└──────────────┘      └──────────────┘      └─────────────┘      └──────┘      └─────┘
```

### company_watcher（企業分析エージェント）

**役割**: 企業の総合分析（リスク情報、ニュース、財務状況など）

**ワークフロー**:
```
┌────────────┐      ┌────────────┐      ┌─────────┐      ┌───────────┐      ┌──────┐      ┌─────┐
│ fetch_info │ ───► │ pre_filter │ ───► │ analyze │ ───► │ summarize │ ───► │ save │ ───► │ END │
│ (yfinance) │      │ (キーワード)│      │ (LLM)   │      │ (総合評価) │      │      │      │     │
└────────────┘      └────────────┘      └─────────┘      └───────────┘      └──────┘      └─────┘
```

### portfolio_manager

**役割**: ポートフォリオを読み込み、LLM で分析・レコメンド

**ワークフロー**:
```
┌──────┐      ┌───────────────┐      ┌─────────┐      ┌──────┐      ┌─────┐
│ load │ ───► │ update_prices │ ───► │ analyze │ ───► │ save │ ───► │ END │
└──────┘      └───────────────┘      └─────────┘      └──────┘      └─────┘
```

### future_insight_agent

**役割**: 収集したニュースから未来シグナルと投資テーマを抽出

**ワークフロー**:
```
┌─────────┐      ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐      ┌──────┐      ┌─────┐
│ collect │ ───► │ extract_signals │ ───► │ synthesize_themes│ ───► │ generate_report │ ───► │ save │ ───► │ END │
└─────────┘      └─────────────────┘      └──────────────────┘      └─────────────────┘      └──────┘      └─────┘
```

### foresight_to_company_translator

**役割**: 未来予測に基づき、重要な企業を特定

**ワークフロー**:
```
┌────────────────┐      ┌──────────────────┐      ┌──────┐      ┌─────┐
│ search_context │ ───► │ analyze_and_find │ ───► │ save │ ───► │ END │
│ (Google News)  │      │ (LLM分析)         │      │      │      │     │
└────────────────┘      └──────────────────┘      └──────┘      └─────┘
```

---

## 6. ツール群

### rss_fetcher.py

**役割**: RSS フィード取得と重複検出

**主要関数**:

| 関数 | 引数 | 戻り値 | 説明 |
|------|------|--------|------|
| `fetch_single_feed` | feed, fetched_ids | `list[NewsItem]` | 単一 RSS フィード取得 |
| `fetch_feeds` | feeds, cache_file | `list[NewsItem]` | 複数フィード取得（重複排除） |
| `load_fetched_ids` | cache_file | `set[str]` | キャッシュ読み込み |
| `save_fetched_ids` | cache_file, ids | None | キャッシュ保存 |

### stock_screener.py

**役割**: FINVIZ を使った株式スクリーニング

**主要関数**:

| 関数 | 引数 | 戻り値 | 説明 |
|------|------|--------|------|
| `screen_movers` | timeframe, min_change, direction, etc. | `list[StockMovement]` | 急騰/急落銘柄検索 |
| `screen_all_performance` | timeframe, max_results | `list[StockMovement]` | 全銘柄パフォーマンス取得 |
| `format_movements` | movements | `str` | 結果フォーマット |

**TimeFrame Enum**:
- `day`, `week`, `month`, `quarter`, `half`, `year`, `ytd`

### portfolio_manager.py

**役割**: ポートフォリオの永続化・価格更新・CSV インポート

**主要関数**:

| 関数 | 引数 | 戻り値 | 説明 |
|------|------|--------|------|
| `load_portfolio` | - | `Portfolio` | ポートフォリオ読み込み |
| `save_portfolio` | portfolio | `Path` | ポートフォリオ保存 |
| `fetch_current_price` | symbol | `float \| None` | 株価取得（yfinance） |
| `update_portfolio_prices` | portfolio | `Portfolio` | 全銘柄価格更新 |
| `import_from_csv` | csv_path, broker | `list[StockHolding]` | CSV インポート |
| `generate_portfolio_report` | portfolio | `str` | レポート生成 |

### company_news_fetcher.py

**役割**: Google News RSS から企業関連ニュースを取得

**主要関数**:

| 関数 | 引数 | 戻り値 | 説明 |
|------|------|--------|------|
| `fetch_company_news` | query, days_back | `list[CompanyNews]` | 企業ニュース取得 |
| `fetch_news_around_date` | query, target_date | `list[CompanyNews]` | 特定日付周辺のニュース取得 |
| `search_topic_news` | topic, keywords | `list[dict]` | トピック検索 |

---

## 7. 外部連携

| 連携先 | URL/API | 用途 |
|--------|---------|------|
| Google Gemini | Gemini API | LLM による分析・フィルタリング |
| yfinance | Yahoo Finance | 株価・企業情報取得 |
| FINVIZ | finvizfinance | 株式スクリーニング |
| Google News RSS | news.google.com | 企業ニュース取得 |
| 各種政府 RSS | 各政府機関 | ニュース収集 |

---

## 8. 出力データ形式

### ニュース（data/*/raw_*.json）

```json
{
  "fetched_at": "2026-01-03T12:00:00",
  "total": 50,
  "items": [
    {
      "id": "abc123",
      "title": "...",
      "source": "whitehouse",
      "url": "...",
      "published": "...",
      "content": "..."
    }
  ]
}
```

### ポートフォリオ（data/portfolio/portfolio.json）

```json
{
  "name": "Main Portfolio",
  "holdings": [
    {
      "symbol": "7203",
      "name": "トヨタ自動車",
      "shares": 100,
      "avg_cost": 2500.0,
      "current_price": 3356.0,
      "account_type": "specific",
      "broker": "manual"
    }
  ],
  "transactions": [],
  "updated_at": "2026-01-03T14:01:18",
  "cash_balance": 0.0
}
```

---

## 9. CLI 使用例

```bash
# インストール
pip install -e .

# ニュース収集（全ソース）
midas collect --source=all

# 米国政府ニュースのみ
midas collect --source=us-gov

# 週間急騰銘柄スクリーニング
midas screen --timeframe=week --min-change=20 --direction=up

# 企業分析（価格イベント + リスク）
midas analyze AAPL --mode=full

# ポートフォリオ管理
midas portfolio show
midas portfolio import --file=holdings.csv
midas portfolio update
midas portfolio analyze

# 未来洞察
midas insight --days=7

# 重要企業発見
midas find-companies "AI will replace software developers"
```

---

## 10. 技術スタック

| カテゴリ | 技術 | 用途 |
|----------|------|------|
| 言語 | Python 3.11+ | メイン言語 |
| エージェント | LangGraph | ワークフロー構築 |
| LLM | Gemini 3 Flash | ニュースフィルタリング・分析 |
| HTTP | httpx | 非同期 HTTP クライアント |
| RSS | feedparser | RSS/Atom パース |
| 株価 | yfinance | 株価・企業情報取得 |
| スクリーニング | finvizfinance | FINVIZ データ取得 |
| データモデル | Pydantic | バリデーション・シリアライズ |
| 設定 | python-dotenv | 環境変数管理 |
| ビルド | Hatchling | パッケージビルド |
| リンター | Ruff | コード品質チェック |

---

## 11. セキュリティ & コンプライアンス

### Credentials Check

| ファイル | 状態 | 説明 |
|----------|------|------|
| `.env` | .gitignore で除外 | API キー格納 |
| `data/` | .gitignore で除外 | ユーザーデータ |

### License Check

- `LICENSE` ファイル: **未設定**

---

## 12. 実装状況

### 実装済み

- ニュース監視エージェント（4種類）
  - 米国政府、技術、その他政府、一般
- 企業分析エージェント
  - 株価イベント分析
  - 企業総合分析（リスク・ニュース・財務）
- ポートフォリオ管理
  - CSV インポート
  - 価格更新
  - LLM 分析
- 未来洞察エージェント
- 重要企業発見エージェント
- 株式スクリーニング（FINVIZ）

### 未実装（Phase 3）

- [ ] 株価急変事例の自動学習
- [ ] 知見のフィードバックループ
