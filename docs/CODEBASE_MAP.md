# Midas Codebase Map

## 1. プロジェクト概要

**プロジェクト名**: Midas

**目的**: 長期投資家向けの投資意思決定支援エージェント群。世界・社会・技術の構造変化を継続的に観測し、長期的に価値が集中・移動する地点を見極める。

**設計思想**:
- 投資対象は「銘柄」ではなく「世の中の変化（構造変化）」
- 価格は結果であり、判断材料の主軸には置かない
- 平常時は何もしない（Hold をデフォルトとする）
- 売却判断は価格ではなく Thesis（前提条件）の破綻によってのみ行う

---

## 2. ファイル構成（Tree View）

```
Midas/
├── .claude/                       # Claude Code 設定
│   ├── commands/
│   │   └── analyze-project.md    # プロジェクト分析コマンド定義
│   └── settings.local.json        # 権限設定
│
├── data/                          # データ保存ディレクトリ（.gitignore）
│   ├── company_analysis/          # 企業分析結果
│   ├── foresights/                # 未来予測データ
│   ├── general/                   # 一般ニュース
│   ├── logs/                      # ログファイル
│   │   ├── agents/                # エージェント個別ログ
│   │   └── runs/                  # 実行単位ログ
│   ├── other_gov/                 # 他国政府ニュース
│   ├── portfolio/                 # ポートフォリオデータ
│   ├── prediction_analysis/       # 予測分析結果
│   ├── prediction_monitor/        # 年次展望監視データ
│   ├── tech/                      # テクノロジーニュース
│   └── us_gov/                    # 米国政府ニュース
│
├── docs/
│   ├── CODEBASE_MAP.md            # このファイル
│   └── requirements.md            # 要件定義書
│
├── src/midas/
│   ├── __init__.py
│   ├── config.py                  # 設定ファイル（API キー、パス等）
│   ├── logging_config.py          # ロギング統合設定
│   ├── main.py                    # CLI エントリーポイント
│   ├── models.py                  # Pydantic データモデル定義
│   ├── orchestrator.py            # 全エージェント統合オーケストレーター
│   │
│   ├── agents/                    # エージェント群
│   │   ├── __init__.py
│   │   ├── news_watcher_base.py  # ニュース監視エージェントの基底クラス
│   │   ├── us_gov_watcher.py     # 米国政府監視エージェント
│   │   ├── tech_news_watcher.py  # テクノロジーニュース監視エージェント
│   │   ├── other_gov_watcher.py  # 他国政府監視エージェント
│   │   ├── general_news_watcher.py # 一般ニュース監視エージェント
│   │   ├── prediction_monitor.py # 年次展望監視エージェント
│   │   ├── foresight_manager.py  # 未来予測管理エージェント
│   │   ├── foresight_to_company_translator.py # 予測→企業変換エージェント
│   │   ├── company_watcher.py    # 企業監視エージェント
│   │   ├── price_event_analyzer.py # 株価イベント分析エージェント
│   │   ├── portfolio_manager.py  # ポートフォリオ管理エージェント
│   │   └── model_calibration_agent.py # モデル校正エージェント
│   │
│   └── tools/                     # ツール群
│       ├── __init__.py
│       ├── rss_fetcher.py         # RSS フィード取得ツール
│       ├── company_news_fetcher.py # 企業ニュース取得ツール
│       ├── stock_screener.py      # 株式スクリーニングツール
│       ├── portfolio_manager.py   # ポートフォリオ管理ツール
│       ├── feedback_loader.py     # フィードバックループツール
│       └── report_generator.py    # レポート生成ツール
│
├── tests/                         # テストスイート
│   ├── __init__.py
│   ├── conftest.py                # pytest 共通フィクスチャ
│   ├── agents/
│   │   └── test_company_watcher.py
│   └── tools/
│
├── pyproject.toml                 # プロジェクト設定・依存関係
└── .gitignore
```

---

## 3. Claude Code 設定（.claude/）

### settings.local.json

**役割**: Claude Code の権限設定

**内容**:
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

MCP（Model Context Protocol）ツールへのアクセス許可を定義。ファイルシステムと Git 操作を許可している。

### commands/analyze-project.md

**役割**: カスタムスラッシュコマンド `/analyze-project` の定義

**機能**: プロジェクト全体を分析し、`docs/CODEBASE_MAP.md` を自動生成するコマンド

---

## 4. コア設定ファイル

### config.py

**役割**: 環境変数とプロジェクト全体の設定を一元管理

**定義される変数/定数**:

| 変数名 | 型 | 説明 |
|--------|-----|------|
| `GEMINI_API_KEY` | `str` | Google Gemini API キー（環境変数から取得） |
| `PROJECT_ROOT` | `Path` | プロジェクトルートディレクトリ |
| `DATA_DIR` | `Path` | データ保存ディレクトリ（`data/`） |
| `NEWS_DIR` | `Path` | ニュースデータディレクトリ（`data/news/`） |
| `LLM_MODEL` | `str` | 使用する LLM モデル名（`gemini-3-flash-preview`） |
| `LLM_MAX_TOKENS` | `int` | LLM 応答の最大トークン数（4096） |

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `extract_llm_text` | `content: Any` | `str` | LLM レスポンスから本文を抽出（Gemini 3 対応） |

---

### logging_config.py

**役割**: プロジェクト全体のロギングを統一管理

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Agent Logger │ ───► │ Main Logger  │ ───► │ File Handler │
│              │      │ (midas)      │      │ - midas.log  │
│              │      │              │      │ - run_*.log  │
│              │      │              │      │ - agent_*.log│
└──────────────┘      └──────────────┘      └──────────────┘
         │                                           │
         │                                           ▼
         └──────────────────────────────────► Console Output
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `setup_main_logger` | `level: int` | `Logger` | メインロガーをセットアップ |
| `get_main_logger` | なし | `Logger` | メインロガーを取得 |
| `get_agent_logger` | `agent_name: str, suffix: str\|None` | `Logger` | エージェント専用ロガーを取得 |
| `get_run_id` | なし | `str` | 現在の実行 ID を取得（YYYYMMDD_HHMMSS） |
| `reset_run_id` | なし | `str` | 新しい実行 ID を生成 |
| `log_agent_start` | `logger, agent_name, initial_state` | なし | エージェント開始ログ |
| `log_agent_end` | `logger, agent_name, final_state, error` | なし | エージェント終了ログ |
| `log_node_start` | `logger, node_name, state` | なし | グラフノード開始ログ |
| `log_node_end` | `logger, node_name, state` | なし | グラフノード終了ログ |
| `log_transition` | `logger, from_node, to_node, condition` | なし | ノード遷移ログ |
| `cleanup_loggers` | なし | なし | 全ロガーをクリーンアップ（セッション終了時） |

**ログ出力先**:
- `data/logs/midas.log` - 全ログの集約
- `data/logs/runs/run_{run_id}.log` - 実行単位のログ
- `data/logs/agents/{agent_name}_{run_id}.log` - エージェント個別ログ

---

## 5. データモデル（models.py）

### ニュース関連モデル

#### NewsCategory (Enum)

ニュースのカテゴリ分類

| 値 | 説明 |
|----|------|
| `LEGISLATION` | 法案 |
| `REGULATION` | 規制 |
| `POLICY` | 政策 |
| `EXECUTIVE_ORDER` | 大統領令 |
| `TRADE` | 通商 |
| `TECHNOLOGY` | 技術政策 |
| `OTHER` | その他 |

#### NewsItem

| フィールド | 型 | 必須 | 説明 |
|------------|-----|------|------|
| `id` | `str` | ✅ | ユニーク ID（URL のハッシュ値） |
| `title` | `str` | ✅ | ニュースタイトル |
| `source` | `str` | ✅ | ソース名（例: 'whitehouse'） |
| `url` | `str` | ✅ | 元記事 URL |
| `published` | `datetime` | ✅ | 公開日時 |
| `content` | `str` | | 本文・概要 |
| `summary` | `str\|None` | | LLM 生成サマリー |
| `is_structural` | `bool` | | 構造変化に関連するか |
| `category` | `NewsCategory\|None` | | ニュースカテゴリ |
| `relevance_reason` | `str\|None` | | 構造変化との関連理由 |

---

### 企業分析モデル

#### RiskInfo

企業のリスク情報

| フィールド | 型 | 必須 | 説明 |
|------------|-----|------|------|
| `category` | `str` | ✅ | カテゴリ（lawsuit/recall/investigation 等） |
| `severity` | `str` | ✅ | 深刻度（low/medium/high/critical） |
| `title` | `str` | ✅ | 問題のタイトル |
| `description` | `str` | ✅ | 詳細説明 |
| `source` | `str` | ✅ | 情報ソース |
| `url` | `str` | ✅ | ソース URL |
| `published` | `datetime` | ✅ | 公開日時 |
| `potential_impact` | `str` | ✅ | 潜在的な影響 |

#### PriceEventAnalysis

株価イベントの分析結果

| フィールド | 型 | 説明 |
|------------|-----|------|
| `event` | `PriceEvent` | 価格イベント情報 |
| `likely_cause` | `str` | 最も可能性の高い原因 |
| `related_news` | `list[CompanyNews]` | 関連ニュース |
| `first_reporter` | `str\|None` | 最初に報道したソース |
| `confidence` | `float` | 分析の確信度（0-1） |

---

### ポートフォリオモデル

#### StockHolding

保有銘柄情報

| フィールド | 型 | 説明 |
|------------|-----|------|
| `symbol` | `str` | ティッカーシンボル |
| `name` | `str` | 企業名 |
| `shares` | `int` | 保有株数 |
| `avg_cost` | `float` | 平均取得単価 |
| `current_price` | `float\|None` | 現在価格 |
| `account_type` | `AccountType` | 口座種別（NISA/特定口座等） |
| `broker` | `str` | 証券会社名 |

**プロパティ**:
- `total_cost` - 総取得コスト
- `current_value` - 現在の評価額
- `unrealized_gain` - 含み損益
- `unrealized_gain_percent` - 含み損益率（%）

---

### 未来予測モデル

#### Foresight

未来予測データ

| フィールド | 型 | 説明 |
|------------|-----|------|
| `id` | `str` | ユニーク ID（例: foresight_20260104_001） |
| `title` | `str` | 予測のタイトル |
| `description` | `str` | 予測の詳細説明 |
| `sources` | `list[ForesightSource]` | 根拠となるソース文書 |
| `created_at` | `datetime` | 作成日時 |

#### FuturePredictionAnalysis

未来予測の分析結果

| フィールド | 型 | 説明 |
|------------|-----|------|
| `prediction` | `str` | 元の予測文 |
| `time_horizon` | `TimeHorizon` | 予測期間（1-2年/3-5年/5-10年） |
| `value_chain_layers` | `list[ValueChainLayer]` | バリューチェーン層の分析 |
| `critical_companies` | `list[CriticalCompany]` | 重要企業リスト |
| `analysis_summary` | `str` | 分析サマリー |
| `investment_implications` | `list[str]` | 投資示唆 |
| `risks` | `list[str]` | リスク要因 |

---

### 学習モデル

#### LearningCase

株価急変事例の学習データ

| フィールド | 型 | 説明 |
|------------|-----|------|
| `id` | `str` | ケース ID |
| `symbol` | `str` | ティッカーシンボル |
| `direction` | `MovementDirection` | 方向（up: 3倍以上/down: 1/3以下） |
| `price_before` | `float` | 変動前価格 |
| `price_after` | `float` | 変動後価格 |
| `change_percent` | `float` | 変動率 |
| `structural_change_type` | `StructuralChangeType\|None` | 構造変化の種類 |
| `root_cause` | `str` | 根本原因 |
| `early_signals` | `list[str]` | 早期警告シグナル |
| `lessons_learned` | `list[str]` | 得られた教訓 |

#### LearnedInsight

学習から得られた知見

| フィールド | 型 | 説明 |
|------------|-----|------|
| `id` | `str` | 知見 ID |
| `title` | `str` | タイトル |
| `category` | `StructuralChangeType` | カテゴリ |
| `description` | `str` | 詳細説明 |
| `detection_patterns` | `list[str]` | 検出パターン |
| `suggested_feeds` | `list[SuggestedFeed]` | 推奨 RSS フィード |
| `suggested_keywords` | `list[SuggestedKeyword]` | 推奨キーワード |
| `target_watchers` | `list[WatcherType]` | 対象 Watcher |

---

## 6. オーケストレーター（orchestrator.py）

**役割**: 全エージェントを統合し、一連のワークフローを実行

**ワークフロー図**:
```
START
  │
  ▼
┌─────────────────┐
│ News Watchers   │ (並列実行)
│ - us_gov        │
│ - tech          │
│ - general       │
│ - other_gov     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Foresight       │ ニュースから未来予測を生成/更新
│ Manager         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Translator      │ 予測から影響を受ける企業を特定
│ (Foresight →    │
│  Company)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Company Watcher │ 企業固有のリスク情報を収集
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Price Analyzer  │ 株価イベントを分析
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Portfolio       │ ポートフォリオを分析し提案を生成
│ Manager         │
└────────┬────────┘
         │
         ▼
        END
```

**State 定義**:
```python
class MidasState(TypedDict):
    news_items: list[NewsItem]           # 収集したニュース
    foresights: list[Foresight]          # 生成された未来予測
    companies: list[str]                 # 特定された企業シンボル
    company_news: dict                   # 企業別ニュース
    portfolio: Portfolio | None          # ポートフォリオデータ
    portfolio_analysis: dict | None      # ポートフォリオ分析結果
    run_id: str                          # 実行 ID
    started_at: str                      # 開始時刻
    completed_at: str | None             # 完了時刻
    error: str | None                    # エラーメッセージ
```

**各ノードの処理**:

| ノード | 処理内容 | 出力 |
|--------|----------|------|
| `news_watchers` | 4つの Watcher を並列実行し、構造変化ニュースを収集 | `news_items` |
| `foresight_manager` | ニュースから未来予測を生成・更新 | `foresights` |
| `translator` | 予測から影響を受ける企業を特定（上位5件まで） | `companies` |
| `company_watcher` | 企業固有のリスク情報を収集（上位3社まで） | `company_news` |
| `price_analyzer` | 株価イベントを分析（上位3社まで） | 解析結果を保存 |
| `portfolio_manager` | ポートフォリオ分析と推奨アクション | `portfolio_analysis` |

---

## 7. エージェント詳細

### 7.1 ニュース監視エージェント群

**基底クラス**: `NewsWatcher` (news_watcher_base.py)

**データフロー**:
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ RSS Feeds   │ ───► │ Fetch News  │ ───► │ Raw JSON    │
└─────────────┘      └─────────────┘      └─────────────┘
                              │
                              ▼
                     ┌─────────────┐
                     │ LLM Filter  │ 構造変化のみ抽出
                     │ (Gemini)    │
                     └──────┬──────┘
                            │
                            ▼
                   ┌─────────────┐
                   │ Filtered    │
                   │ News JSON   │
                   └─────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `fetch_news` | `state: WatcherState` | `WatcherState` | RSS フィードからニュースを取得 |
| `filter_news` | `state: WatcherState` | `WatcherState` | LLM で構造変化ニュースをフィルタリング |
| `save_results` | `state: WatcherState` | `WatcherState` | フィルタ済み結果を JSON 保存 |
| `run` | なし | `WatcherState` | 完全なワークフローを実行 |

**派生クラス**:

| エージェント | ファイル | データソース |
|------------|----------|------------|
| `us_gov_watcher` | us_gov_watcher.py | White House, Congress, Federal Register, SEC, USTR |
| `tech_news_watcher` | tech_news_watcher.py | Ars Technica, TechCrunch, Wired, MIT Tech Review |
| `other_gov_watcher` | other_gov_watcher.py | EU, UK, 中国, 日本, IMF, World Bank |
| `general_news_watcher` | general_news_watcher.py | Yahoo Finance, Bloomberg, Reuters, CNBC |

---

### 7.2 prediction_monitor（年次展望監視エージェント）

**役割**: McKinsey、BCG、WEF などの年次展望記事をスキャンし、中長期の社会変化トレンドを検出

**データフロー**:
```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Google News │ ───► │ Search      │ ───► │ Articles    │
│ RSS         │      │ "{year}     │      │ List        │
│             │      │  outlook"   │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
                              │
                              ▼
                     ┌─────────────┐
                     │ LLM Extract │ トレンド抽出
                     │ Trends      │
                     └──────┬──────┘
                            │
                            ▼
                   ┌─────────────┐
                   │ Report JSON │
                   └─────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_scan` | `year: int` | `dict` | 年次展望記事をスキャン・分析 |
| `expand_sources` | なし | `list[dict]` | AI によるソース拡張提案 |
| `load_sources` | なし | `dict` | 現在のソースリストを読み込み |
| `format_report` | `result: dict` | `str` | レポートをフォーマット |

**出力データ形式**:
```json
{
  "year": 2026,
  "analyzed_at": "2026-01-04T15:00:00",
  "articles": [
    {
      "title": "2026 Outlook: AI and Automation",
      "source": "McKinsey",
      "url": "https://...",
      "published": "2025-12-01T00:00:00"
    }
  ],
  "trends": [
    {
      "category": "technology_shift",
      "title": "AI エージェント化の加速",
      "description": "...",
      "time_horizon": "1-3 years",
      "confidence": 0.85
    }
  ]
}
```

**保存先**: `data/prediction_monitor/reports/prediction_monitor_{year}_{timestamp}.json`

---

### 7.3 foresight_manager（未来予測管理エージェント）

**役割**: 未来予測の管理オーケストレーター。2つのモードで動作:

**モード判定**:
```
┌─────────────────────────────────┐
│ 最終フル更新から 90 日以上？    │
│ または force_full=True？        │
└────────┬────────────────┬───────┘
         │YES             │NO
         ▼                ▼
    ┌────────┐      ┌────────────┐
    │ Full   │      │ Incremental│
    │ Mode   │      │ Mode       │
    └────────┘      └────────────┘
```

**Full Mode データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ prediction_  │ ───► │ LLM          │ ───► │ Foresights   │
│ monitor      │      │ Generate     │      │ Generated    │
│ (年次展望)   │      │ Foresights   │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
```

**Incremental Mode データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Existing     │      │ Recent News  │      │ LLM Adjust   │
│ Foresights   │ ───► │ from         │ ───► │ Foresights   │
│              │      │ Watchers     │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_agent` | `force_full: bool` | `dict` | エージェント実行 |
| `determine_mode` | `force_full: bool` | `str` | モード判定（"full" or "incremental"） |
| `load_foresights` | なし | `list[Foresight]` | 既存 foresight を読み込み |
| `save_foresights` | `foresights, is_full_update` | `Path` | foresight を保存 |
| `load_watcher_news` | `days_back: int` | `list[NewsItem]` | 過去 N 日のニュースを読み込み |

**保存先**: `data/foresights/foresights.json`

---

### 7.4 foresight_to_company_translator（展望-企業変換エージェント）

**役割**: 未来予測から重要企業を特定。3段階 LLM 分析を実行。

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Prediction   │ ───► │ LLM Step 1   │ ───► │ Value Chain  │
│              │      │ Decompose    │      │ Layers       │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ LLM Step 2   │
                     │ Identify     │ ───► Critical Companies
                     │ Companies    │
                     └──────┬───────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ LLM Step 3   │
                   │ Analyze      │ ───► Bottlenecks
                   │ Bottlenecks  │
                   └──────────────┘
```

**処理ステップ**:

1. **バリューチェーン分解**: 予測を実現するためのバリューチェーン層を特定（完成品、コア部品、素材、製造装置、インフラ、ソフトウェア）
2. **企業特定**: 各層で重要な役割を果たす企業をリストアップ（市場リーダー、挑戦者、ニッチプレイヤー、新興企業）
3. **ボトルネック分析**: 各層のボトルネック度を評価（supply/tech/regulation）

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_agent` | `prediction: str` | `dict` | 予測分析を実行 |
| `format_analysis` | `result: dict` | `str` | 分析結果をフォーマット |

**出力例**:
```json
{
  "prediction": "電気自動車が主流になる",
  "value_chain_layers": [
    {
      "name": "バッテリーセル",
      "layer_type": "core_component",
      "bottleneck_level": "high",
      "bottleneck_reason": "供給制約と技術的課題"
    }
  ],
  "critical_companies": [
    {
      "name": "CATL",
      "symbol": "300750.SZ",
      "role": "バッテリーセル製造",
      "layer": "core_component",
      "market_position": "leader",
      "confidence": 0.9
    }
  ]
}
```

**保存先**: `data/prediction_analysis/prediction_{title}_{timestamp}.json`

---

### 7.5 company_watcher（企業監視エージェント）

**役割**: 企業固有のリスク要因を監視（訴訟、リコール、調査、経営者交代、規制等）

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Company News │ ───► │ Risk Keyword │ ───► │ Candidate    │
│ Fetcher      │      │ Filter       │      │ News         │
│ (Google News)│      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ LLM Batch    │
                     │ Analysis     │ ───► RiskInfo List
                     │              │
                     └──────────────┘
```

**リスクキーワード**:
- lawsuit, sued, investigation, recall, downgrade
- fraud, scandal, bankruptcy, restructuring
- CEO resign, management change, regulatory action

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_agent` | `symbol: str` | `dict` | 企業監視を実行 |
| `format_results` | `result: dict` | `str` | 結果をフォーマット |

**出力データ形式**:
```json
{
  "symbol": "AAPL",
  "name": "Apple Inc.",
  "analyzed_at": "2026-01-04T15:00:00",
  "risk_info": [
    {
      "category": "lawsuit",
      "severity": "medium",
      "title": "反トラスト訴訟",
      "description": "...",
      "source": "Reuters",
      "url": "https://...",
      "potential_impact": "..."
    }
  ],
  "risk_summary": "..."
}
```

**保存先**: `data/company_analysis/risk_info_{symbol}_{timestamp}.json`

---

### 7.6 price_event_analyzer（株価イベント分析エージェント）

**役割**: 過去1ヶ月の株価変動を分析し、原因となったニュースを特定

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ yfinance     │ ───► │ Detect       │ ───► │ Price Events │
│ (株価データ) │      │ Events       │      │ List         │
│              │      │ (±20% jump)  │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ Fetch News   │
                     │ Around Event │ ───► Related News
                     │              │
                     └──────┬───────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ LLM Analyze  │
                   │ Causality    │ ───► PriceEventAnalysis
                   └──────────────┘
```

**イベント検出基準**:
- 1日で ±20% 以上の価格変動
- 出来高が平均の 2倍以上

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_agent` | `symbol: str` | `dict` | 株価イベント分析を実行 |
| `format_analysis` | `result: dict` | `str` | 分析結果をフォーマット |

**保存先**: `data/company_analysis/price_analysis_{symbol}_{timestamp}.json`

---

### 7.7 portfolio_manager（ポートフォリオ管理エージェント）

**役割**: ポートフォリオを分析し、保有継続・売却検討の提案を行う

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Portfolio    │ ───► │ Update Prices│ ───► │ Current      │
│ JSON         │      │ (yfinance)   │      │ Valuation    │
│              │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ LLM Analyze  │
                     │ Holdings     │ ───► Recommendations
                     │              │
                     └──────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_agent` | なし | `dict` | ポートフォリオ分析を実行 |

**保存先**: `data/portfolio/analysis_{timestamp}.json`

---

### 7.8 model_calibration_agent（モデル校正エージェント）

**役割**: 株価急変事例から学習し、他エージェントの監視精度を改善

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Stock        │ ───► │ Find 3x or   │ ───► │ Cases List   │
│ Screener     │      │ 1/3 Movers   │      │              │
│              │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ Analyze Each │
                     │ Case         │ ───► LearningCase
                     │              │
                     └──────┬───────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ Generate     │
                   │ Insights     │ ───► LearnedInsight
                   │              │      (SuggestedFeed等)
                   └──────────────┘
```

**学習ロジック**:

| 変動方向 | 学習内容 |
|---------|---------|
| 3倍上昇 | foresight でカバーできているか確認。未カバーなら news_watcher に新規 RSS/キーワード追加 |
| 1/3下落 | company_watcher がリスクを検出できていたか確認。未検出なら監視強化 |

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `run_agent` | `period: str, max_cases: int` | `dict` | 学習を実行 |
| `list_insights` | なし | `list[LearnedInsight]` | 蓄積した知見を取得 |
| `list_cases` | なし | `list[LearningCase]` | 分析済みケースを取得 |
| `format_report` | `result: dict` | `str` | レポートをフォーマット |

---

## 8. ツール詳細

### 8.1 rss_fetcher.py

**役割**: RSS フィードから新着記事を取得（重複排除機能付き）

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ RSS Feed URL │ ───► │ feedparser   │ ───► │ Parsed Items │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ Check Cache  │ 重複チェック
                     │ (fetched_ids)│
                     └──────┬───────┘
                            │
                            ▼
                   ┌──────────────┐
                   │ New Items    │
                   │ Only         │
                   └──────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `fetch_feeds` | `feeds: list[dict], cache_file: Path` | `list[NewsItem]` | 複数フィードから新着記事を取得 |
| `fetch_single_feed` | `feed: dict, fetched_ids: set` | `list[NewsItem]` | 単一フィードから取得 |
| `load_fetched_ids` | `cache_file: Path` | `set[str]` | キャッシュから既読 ID を読み込み |
| `save_fetched_ids` | `cache_file: Path, ids: set` | なし | キャッシュに ID を保存 |

**キャッシュ形式**:
```json
{
  "ids": ["abc123...", "def456..."],
  "updated_at": "2026-01-04T15:00:00"
}
```

**エラーハンドリング**:
- HTTP エラー → 該当フィードをスキップ
- XML パースエラー → 該当フィードをスキップ
- 続行可能なエラーはログに記録

---

### 8.2 company_news_fetcher.py

**役割**: Google News RSS API を使って企業固有のニュースを取得

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Company      │ ───► │ Build Google │ ───► │ RSS Feed URL │
│ Symbol       │      │ News Query   │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼
                     ┌──────────────┐
                     │ feedparser   │ ───► CompanyNews List
                     └──────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `fetch_company_news` | `symbol: str, days_back: int` | `list[CompanyNews]` | 企業ニュースを取得 |

**クエリ構築**:
- `{company_name} OR {symbol}` で検索
- 過去 N 日に限定（`when:{days_back}d`）

---

### 8.3 stock_screener.py

**役割**: FINVIZ API を使って株価急変銘柄をスクリーニング

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ FINVIZ API   │ ───► │ Filter       │ ───► │ Movers List  │
│ (screener)   │      │ By Change %  │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `screen_movers` | `timeframe, min_change, direction, sector` | `list[StockMovement]` | 株価変動銘柄をスクリーニング |
| `format_movements` | `movements: list[StockMovement]` | `str` | 結果をテーブル形式でフォーマット |

**スクリーニング条件**:
- 期間: day/week/month/quarter/half/year/ytd
- 最小変動率: デフォルト 20%
- 方向: up/down/both
- セクター: オプション

---

### 8.4 portfolio_manager.py (tool)

**役割**: ポートフォリオデータの CRUD 操作

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `load_portfolio` | なし | `Portfolio` | ポートフォリオを読み込み |
| `save_portfolio` | `portfolio: Portfolio` | `Path` | ポートフォリオを保存 |
| `add_holdings_to_portfolio` | `portfolio, holdings` | `Portfolio` | 保有銘柄を追加 |
| `update_portfolio_prices` | `portfolio: Portfolio` | `Portfolio` | 株価を更新（yfinance） |
| `import_from_csv` | `csv_path: str, broker: str` | `list[StockHolding]` | CSV からインポート |
| `generate_portfolio_report` | `portfolio: Portfolio` | `str` | レポート生成 |

---

### 8.5 feedback_loader.py

**役割**: model_calibration_agent からのフィードバックを読み込み、動的に RSS フィード/キーワードを追加

**データフロー**:
```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ LearnedInsight│ ───► │ Extract      │ ───► │ Dynamic Feeds│
│ (suggested_  │      │ Suggested    │      │ + Keywords   │
│  feeds)      │      │ Feeds        │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
```

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `build_dynamic_feeds` | `base_feeds, watcher_type` | `list[dict]` | 基本フィード + 動的フィードを結合 |
| `format_feedback_summary` | `watcher_type: WatcherType` | `str` | フィードバックサマリーを生成 |

---

### 8.6 report_generator.py

**役割**: 分析結果を JSON/Markdown で保存

**主要関数**:

| 関数名 | 引数 | 戻り値 | 説明 |
|--------|------|--------|------|
| `save_report` | `data: dict, filename: str, category: str` | `Path` | JSON レポートを保存 |
| `generate_foresight_report` | `foresights: list` | `str` | Foresight レポートを生成 |

---

## 9. CLI コマンド（main.py）

### コマンド一覧

```bash
# ニュース収集
midas collect --source all          # 全 Watcher 実行
midas collect --source us-gov       # 米国政府のみ
midas collect --source tech         # テクノロジーのみ

# 株式スクリーニング
midas screen --timeframe week --min-change 20 --direction both

# 企業分析
midas analyze AAPL --mode full      # 価格イベント + リスク分析
midas analyze AAPL --mode price     # 価格イベントのみ
midas analyze AAPL --mode risk      # リスクのみ

# ポートフォリオ管理
midas portfolio show                # 保有銘柄一覧
midas portfolio import --file portfolio.csv --broker nomura
midas portfolio add AAPL 100 150.0 --name "Apple Inc."
midas portfolio update              # 株価更新
midas portfolio analyze             # LLM 分析

# 未来予測
midas foresight scan                # 自動モード判定
midas foresight scan --force-full   # フル更新強制
midas foresight list                # 全 foresight 一覧

# 年次展望監視
midas prediction-monitor scan --year 2026
midas prediction-monitor sources    # ソース一覧
midas prediction-monitor expand     # AI によるソース拡張提案

# 企業発見
midas find-companies "電気自動車が主流になる"

# 学習
midas learn scan --period month --max-cases 20
midas learn insights                # 蓄積した知見
midas learn cases                   # 分析済み事例

# オーケストレーター（全実行）
midas run
```

---

## 10. 技術スタック

| カテゴリ | 技術 | 用途 |
|----------|------|------|
| 言語 | Python 3.11+ | メイン言語 |
| フレームワーク | LangGraph | エージェント構築・ワークフロー管理 |
| LLM | Google Gemini API | ニュースフィルタリング、分析、知見抽出 |
| 株価取得 | yfinance | Yahoo Finance からの株価データ取得 |
| 株式スクリーニング | finvizfinance | FINVIZ スクリーナー |
| HTTP クライアント | httpx | 非同期 HTTP リクエスト |
| RSS パーサー | feedparser | RSS フィードのパース |
| データモデル | Pydantic | 型安全なデータモデル |
| データ保存 | JSON ファイル | ファイルベースの永続化 |
| テスト | pytest, pytest-asyncio | ユニットテスト |
| コード品質 | ruff | リンター・フォーマッター |

---

## 11. セキュリティ & コンプライアンス

### Credentials Check

| ファイル | 状態 | 説明 |
|----------|------|------|
| `.env` | ✅ .gitignore で除外 | GEMINI_API_KEY を格納 |
| `data/` | ✅ .gitignore で除外 | 全ての実行時データ |

### License Check

プロジェクトルートに `LICENSE` ファイルは現在未設置。

---

## 12. 使い方

### インストール

```bash
# リポジトリクローン
git clone <repository-url>
cd Midas

# 仮想環境作成
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存関係インストール
pip install -e .

# 開発用依存関係（テスト用）
pip install -e ".[dev]"
```

### 環境変数設定

```bash
# .env ファイルを作成
cat > .env << EOF
GEMINI_API_KEY=your_api_key_here
EOF
```

### 基本的な実行フロー

```bash
# 1. ニュース収集（毎日実行推奨）
midas collect --source all

# 2. 未来予測の更新（週次実行推奨）
midas foresight scan

# 3. ポートフォリオ分析（必要時）
midas portfolio analyze

# 4. 全ワークフロー実行（月次実行推奨）
midas run
```

### テスト実行

```bash
# 全テスト実行
pytest

# カバレッジ付き
pytest --cov=midas --cov-report=html

# 特定のテストのみ
pytest tests/agents/test_company_watcher.py
```

---

## 13. 実装状況

### 実装済み ✅

**Phase 1: MVP**
- プロジェクト構造のセットアップ
- 株価データ収集（yfinance）
- 株式スクリーニング（FINVIZ）
- 基本的な企業分析機能
- CLI でのレポート出力

**Phase 2: ニュース監視**
- ニュース収集エージェント（米国政府）
- ニュース収集エージェント（新技術）
- ニュース収集エージェント（その他政府）
- ニュース収集エージェント（一般ニュース）
- 情報選別フィルタリング（LLM）
- ポートフォリオ管理機能
- 未来洞察エージェント
- 重要企業発見エージェント

**Phase 3: 学習・改善**
- 株価急変事例の収集・分析
- 知見のフィードバックループ
- 統合ロギングシステム
- 統合オーケストレーター

### 未実装（Phase 3 残タスク）

- 通知機能（Slack / LINE）
- PostgreSQL でのデータ永続化
- Google Slides への自動レポート出力（実装途中）
- Web UI ダッシュボード

---

## 14. アーキテクチャ概要図

```
                         ┌──────────────────────────────────┐
                         │     Midas Orchestrator           │
                         │  (orchestrator.py)               │
                         └──────────┬───────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ News Watchers    │   │ Foresight        │   │ Portfolio        │
│ (並列実行)       │   │ Manager          │   │ Manager          │
│                  │   │                  │   │                  │
│ - us_gov         │──►│ - Full Mode      │──►│ - 保有銘柄分析   │
│ - tech           │   │ - Incremental    │   │ - リスク検出     │
│ - general        │   │   Mode           │   │ - 推奨アクション │
│ - other_gov      │   │                  │   │                  │
└──────────────────┘   └────────┬─────────┘   └──────────────────┘
                                │
                                ▼
                    ┌──────────────────┐
                    │ Foresight-to-    │
                    │ Company          │
                    │ Translator       │
                    │                  │
                    │ - Value Chain    │
                    │ - Company List   │
                    │ - Bottlenecks    │
                    └────────┬─────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Company      │  │ Price Event  │  │ Model        │
│ Watcher      │  │ Analyzer     │  │ Calibration  │
│              │  │              │  │ Agent        │
│ - Risk Info  │  │ - Price Jump │  │              │
│ - News Scan  │  │ - Causality  │  │ - Learning   │
│              │  │   Analysis   │  │ - Feedback   │
└──────────────┘  └──────────────┘  └──────┬───────┘
                                            │
                                            │ フィードバック
                                            │ (RSS/Keyword追加)
                                            │
                      ┌─────────────────────┴─────────────┐
                      │                                   │
                      ▼                                   ▼
           ┌──────────────────┐              ┌──────────────────┐
           │ News Watchers    │              │ Company Watcher  │
           │ (動的フィード追加)│              │ (監視強化)       │
           └──────────────────┘              └──────────────────┘
```

---

## 15. 開発ガイドライン

### 新しいエージェントの追加

1. `src/midas/agents/` に新しいファイルを作成
2. 必要に応じて `models.py` に新しいデータモデルを追加
3. `src/midas/agents/__init__.py` にエクスポート追加
4. `main.py` に CLI コマンドを追加
5. `orchestrator.py` にノードとして統合（必要に応じて）
6. テストを `tests/agents/` に追加

### 新しいツールの追加

1. `src/midas/tools/` に新しいファイルを作成
2. `src/midas/tools/__init__.py` にエクスポート追加
3. テストを `tests/tools/` に追加

### コーディング規約

- 型ヒントを必ず使用（`mypy` 準拠を目指す）
- Pydantic モデルでデータ構造を定義
- ロガーは `logging_config.get_agent_logger()` で取得
- エラーハンドリングは適切に実施（ログに記録し、続行可能なら続行）
- ファイル保存時はタイムスタンプを含める（YYYYMMDD_HHMMSS）

---

**最終更新**: 2026-01-04
