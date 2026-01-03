# Midas コードベースマップ

## プロジェクト概要

**Midas** は、長期投資家向けの投資意思決定支援エージェントシステム。世界・社会・技術の構造変化を継続的に監視して、価値が移動する地点を見極める。

### 設計思想

- 投資対象は「銘柄」ではなく「世の中の変化（構造変化）」
- 価格は結果であり、判断材料の主軸には置かない
- 平常時は Hold をデフォルト
- 売却判断は Thesis（前提条件）の破綻によってのみ行う

---

## ファイル構成

```
Midas/
├── .claude/                      # Claude Code 設定ディレクトリ
│   ├── commands/                 # カスタムスラッシュコマンド定義
│   │   └── analyze-project.md   # /analyze-project コマンド
│   └── settings.local.json      # プロジェクト固有の権限設定
├── data/
│   └── news/                     # ニュースデータ保存ディレクトリ（JSON）
├── docs/
│   ├── requirements.md           # 要件定義書（投資哲学・機能要件）
│   └── CODEBASE_MAP.md           # コードベースマップ（本ファイル）
├── src/
│   └── midas/
│       ├── __init__.py           # パッケージ初期化、バージョン定義
│       ├── __main__.py           # python -m midas エントリーポイント
│       ├── main.py               # CLI メインエントリーポイント
│       ├── config.py             # 設定・環境変数・RSS ソース定義
│       ├── models.py             # Pydantic データモデル
│       ├── agents/
│       │   ├── __init__.py
│       │   └── us_gov_news.py    # 米国政府ニュース収集エージェント
│       └── tools/
│           ├── __init__.py
│           └── rss_fetcher.py    # RSS フィード取得ツール
├── .env                          # 環境変数（GEMINI_API_KEY）※Git管理外
├── .env.example                  # 環境変数のテンプレート
├── .gitignore                    # Git 管理除外設定
└── pyproject.toml                # プロジェクト設定・依存関係
```

---

## Claude Code 設定 (.claude/)

Claude Code（AI コーディングアシスタント）のプロジェクト固有設定を格納するディレクトリ。

### .claude/settings.local.json

プロジェクト固有の権限設定ファイル。確認なしで実行できる MCP ツールを定義。

```json
{
  "permissions": {
    "allow": [
      "mcp__filesystem__list_directory",  // ディレクトリ一覧取得
      "mcp__git__git_status",             // Git ステータス確認
      "mcp__git__git_log",                // Git ログ確認
      "mcp__filesystem__directory_tree"   // ディレクトリツリー取得
    ]
  }
}
```

### .claude/commands/analyze-project.md

`/analyze-project` カスタムスラッシュコマンドの定義ファイル。このコマンドを実行すると、プロジェクト構造をスキャンして `docs/CODEBASE_MAP.md` を更新する。

**コマンドの役割**:
- プロジェクトのディレクトリツリーを出力
- 各ファイルの役割を表形式でまとめ
- セキュリティ・コンプライアンスチェック
- 変更点の追跡

---

## 各ファイルの詳細解説

### pyproject.toml

**役割**: プロジェクトのメタデータ、依存関係、ビルド設定を定義

| 設定項目 | 値 | 説明 |
|----------|-----|------|
| `name` | `midas` | パッケージ名 |
| `version` | `0.1.0` | 現在のバージョン |
| `requires-python` | `>=3.11` | Python 3.11 以上が必要 |
| `build-backend` | `hatchling` | ビルドツール |

**依存ライブラリ**:
| ライブラリ | バージョン | 用途 |
|-----------|-----------|------|
| `langgraph` | >= 0.2 | エージェントワークフロー構築 |
| `langchain-google-genai` | >= 2.0 | Gemini API 連携 |
| `feedparser` | >= 6.0 | RSS フィードパース |
| `httpx` | >= 0.27 | 非同期 HTTP クライアント |
| `python-dotenv` | >= 1.0 | 環境変数読み込み |
| `pydantic` | >= 2.0 | データバリデーション |

---

### src/midas/config.py

**役割**: アプリケーション全体の設定を一元管理

**データフロー**:
```
.env ファイル → dotenv.load_dotenv() → 環境変数 → config モジュール
```

**定義内容**:

| 変数 | 型 | 説明 |
|------|-----|------|
| `GEMINI_API_KEY` | `str \| None` | Gemini API キー（.env から読み込み） |
| `PROJECT_ROOT` | `Path` | プロジェクトルートディレクトリ |
| `DATA_DIR` | `Path` | データ保存ディレクトリ（`data/`） |
| `NEWS_DIR` | `Path` | ニュース保存ディレクトリ（`data/news/`） |
| `RSS_SOURCES` | `dict` | RSS ソース定義（後述） |
| `LLM_MODEL` | `str` | 使用する LLM モデル名（`gemini-2.5-flash`） |
| `LLM_MAX_TOKENS` | `int` | 最大トークン数（4096） |

**RSS_SOURCES の構造**:
```python
RSS_SOURCES = {
    "source_key": {
        "name": "表示名",
        "url": "RSS フィード URL",
        "description": "ソースの説明"
    }
}
```

**登録済み RSS ソース**:
| キー | 名前 | URL | 説明 |
|------|------|-----|------|
| `whitehouse` | White House | `https://www.whitehouse.gov/feed/` | ホワイトハウス公式発表 |
| `congress_bills` | Congress - Bills | `https://www.congress.gov/rss/bill-status-all.xml` | 議会法案ステータス |
| `federal_register` | Federal Register | `https://www.federalregister.gov/documents/current.rss` | 連邦規則・公告 |
| `sec_news` | SEC News | `https://www.sec.gov/news/pressreleases.rss` | SEC プレスリリース |
| `ustr` | US Trade Representative | `https://ustr.gov/.../press-releases/rss` | 通商代表部発表 |

---

### src/midas/models.py

**役割**: アプリケーション全体で使用するデータ構造を Pydantic モデルで定義

#### NewsCategory (Enum)

ニュースの分類カテゴリ。LLM がニュースを分析する際に使用。

| 値 | 説明 | 例 |
|----|------|-----|
| `legislation` | 法案 | 新しい法律の制定・改正 |
| `regulation` | 規制 | 規制当局による新ルール |
| `policy` | 政策 | 政府の政策発表 |
| `executive_order` | 大統領令 | 大統領による行政命令 |
| `trade` | 通商 | 関税、貿易協定 |
| `technology` | 技術政策 | 技術規制、産業政策 |
| `other` | その他 | 上記に該当しないもの |

#### NewsItem

単一のニュース記事を表すモデル。

| フィールド | 型 | 必須 | 説明 |
|------------|-----|------|------|
| `id` | `str` | ✅ | URL の MD5 ハッシュ（先頭12文字） |
| `title` | `str` | ✅ | 記事タイトル |
| `source` | `str` | ✅ | ソースキー（例: `whitehouse`） |
| `url` | `str` | ✅ | 元記事の URL |
| `published` | `datetime` | ✅ | 公開日時 |
| `content` | `str` | - | 記事本文または概要 |
| `summary` | `str \| None` | - | LLM 生成の要約（将来実装） |
| `is_structural` | `bool` | - | 構造変化に関連するか（デフォルト: False） |
| `category` | `NewsCategory \| None` | - | ニュースカテゴリ |
| `relevance_reason` | `str \| None` | - | LLM が判定した関連性の理由 |

#### NewsCollection

ニュースのコレクション。バッチ処理時に使用（現在未使用）。

---

### src/midas/tools/rss_fetcher.py

**役割**: RSS フィードを取得して `NewsItem` オブジェクトに変換するツール

**データフロー**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                         rss_fetcher.py                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  config.RSS_SOURCES                                                 │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    HTTP GET     ┌──────────────┐                   │
│  │ source_key  │ ──────────────► │  RSS URL     │                   │
│  │ (whitehouse)│    (httpx)      │ (外部サーバー)│                   │
│  └─────────────┘                 └──────┬───────┘                   │
│                                         │                           │
│                                         ▼ XML/RSS                   │
│                                  ┌──────────────┐                   │
│                                  │  feedparser  │                   │
│                                  │  .parse()    │                   │
│                                  └──────┬───────┘                   │
│                                         │                           │
│                                         ▼ feed.entries              │
│                                  ┌──────────────┐                   │
│                                  │ 各エントリを │                   │
│                                  │ NewsItem に  │                   │
│                                  │ 変換         │                   │
│                                  └──────┬───────┘                   │
│                                         │                           │
│                                         ▼                           │
│                               list[NewsItem]                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**主要関数**:

| 関数 | 引数 | 戻り値 | 説明 |
|------|------|--------|------|
| `_generate_id(url)` | `str` | `str` | URL から一意の ID を生成（MD5 先頭12文字） |
| `_parse_published(entry)` | `dict` | `datetime` | RSS エントリから公開日時を抽出 |
| `_parse_content(entry)` | `dict` | `str` | RSS エントリから本文を抽出（content > summary > description の優先順） |
| `fetch_rss(source_key)` | `str` | `list[NewsItem]` | 単一ソースから RSS を取得 |
| `fetch_all_sources()` | なし | `list[NewsItem]` | 全ソースから RSS を取得して結合 |

**エラーハンドリング**:
- HTTP エラー: `httpx.HTTPStatusError` を発生
- 日付パースエラー: 現在時刻をフォールバック
- 個別ソースのエラー: ログ出力して継続（他ソースは処理続行）

---

### src/midas/agents/us_gov_news.py

**役割**: LangGraph を使った米国政府ニュース収集・フィルタリングエージェント

**データフロー**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                       us_gov_news.py                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐      ┌──────────┐      ┌────────┐      ┌─────┐        │
│  │  fetch  │ ───► │  filter  │ ───► │  save  │ ───► │ END │        │
│  └────┬────┘      └────┬─────┘      └────┬───┘      └─────┘        │
│       │                │                 │                          │
│       ▼                ▼                 ▼                          │
│  rss_fetcher     Gemini LLM        JSON ファイル                    │
│  .fetch_all()    で判定            data/news/                       │
│                                    us_gov_YYYYMMDD_HHMMSS.json     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**AgentState（状態管理）**:
```python
class AgentState(TypedDict):
    raw_items: list[NewsItem]      # fetch で取得した全ニュース
    filtered_items: list[NewsItem] # filter で抽出した構造変化ニュース
    saved_path: str | None         # save で保存したファイルパス
    error: str | None              # エラーメッセージ（あれば）
```

**各ノードの処理**:

#### 1. fetch_news（取得ノード）
- **入力**: 空の `AgentState`
- **処理**: `rss_fetcher.fetch_all_sources()` を呼び出し
- **出力**: `raw_items` にニュースリストをセット
- **エラー時**: `error` にメッセージをセット

#### 2. filter_news（フィルタリングノード）
- **入力**: `raw_items` に格納されたニュースリスト
- **処理**:
  1. 各ニュースを Gemini LLM に送信
  2. システムプロンプトで「構造変化」の判定基準を指示
  3. JSON 形式で `is_structural`, `category`, `reason` を取得
  4. `is_structural: true` のニュースのみ抽出
- **出力**: `filtered_items` にフィルタ済みリストをセット

**LLM への指示（システムプロンプト要約）**:
- 構造変化として判定するもの:
  - 技術の実用化・廃止
  - 規制・政策変更
  - 通商政策変更
  - 政府の産業政策
  - 競争環境の変化
- 無視するもの:
  - 短期的な市場変動
  - 四半期決算
  - 人事異動（大企業 CEO 除く）
  - ルーティンの規制申請
  - オピニオン記事

#### 3. save_results（保存ノード）
- **入力**: `filtered_items`
- **処理**:
  1. タイムスタンプ付きファイル名を生成
  2. JSON 形式でシリアライズ
  3. `data/news/us_gov_YYYYMMDD_HHMMSS.json` に保存
- **出力**: `saved_path` にファイルパスをセット

**保存される JSON 構造**:
```json
{
  "fetched_at": "2024-01-01T12:00:00",
  "total_raw": 150,
  "total_filtered": 12,
  "items": [
    {
      "id": "abc123def456",
      "title": "...",
      "source": "whitehouse",
      "url": "...",
      "published": "...",
      "content": "...",
      "is_structural": true,
      "category": "policy",
      "relevance_reason": "..."
    }
  ]
}
```

---

### src/midas/main.py

**役割**: CLI エントリーポイント。ユーザーからのコマンドを受け付けて処理を実行

**データフロー**:
```
┌─────────────────────────────────────────────────────────────────────┐
│                           main.py                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  $ midas collect --source=us-gov                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐                                                    │
│  │ argparse    │  コマンドライン引数をパース                         │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐                                                    │
│  │ print_banner│  バナー表示                                         │
│  └──────┬──────┘                                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ collect_us_news()│  us_gov_news.run_agent() を呼び出し           │
│  └──────┬───────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  結果を整形して標準出力に表示                                        │
│  - フィルタされたニュース数                                          │
│  - 各ニュースのタイトル・カテゴリ・理由                              │
│  - 保存先ファイルパス                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**CLI コマンド**:
```bash
# 米国政府ニュース収集
midas collect --source=us-gov

# ヘルプ表示
midas --help
midas collect --help
```

**出力例**:
```
╔═══════════════════════════════════════════════════════════════╗
║  Midas - Investment Decision Support Agent                   ║
║  "Identifying structural changes for long-term investors"    ║
╚═══════════════════════════════════════════════════════════════╝

Starting US Government News Collection...
--------------------------------------------------
Fetching news from RSS sources...
  Fetched 50 items from whitehouse
  Fetched 100 items from congress_bills
  ...
Filtering news with LLM...
  [+] New Trade Policy Announced...
  [+] Technology Export Controls Updated...
Filtered: 12 structural news items
Saving results...
Saved to: data/news/us_gov_20240101_120000.json
--------------------------------------------------

=== Found 12 structural news items ===

[policy] New Trade Policy Announced
    Reason: Major shift in trade relations affecting supply chains
    URL: https://...
```

---

### src/midas/__main__.py

**役割**: `python -m midas` で実行可能にするためのエントリーポイント

```python
from midas.main import main

if __name__ == "__main__":
    main()
```

---

### src/midas/__init__.py

**役割**: パッケージ初期化。バージョン情報を定義。

```python
__version__ = "0.1.0"
```

---

## 技術スタック

| カテゴリ | 技術 | 用途 |
|----------|------|------|
| 言語 | Python 3.11+ | メイン言語 |
| エージェント | LangGraph | ワークフロー構築 |
| LLM | Gemini 2.5 Flash | ニュースフィルタリング |
| HTTP | httpx | 非同期 HTTP リクエスト |
| RSS | feedparser | RSS/Atom パース |
| データモデル | Pydantic | バリデーション・シリアライズ |
| 設定 | python-dotenv | 環境変数管理 |
| ビルド | Hatchling | パッケージビルド |
| リンター | Ruff | コード品質チェック |

---

## セキュリティ & コンプライアンス

### Credentials Check

| ファイル | 状態 | 説明 |
|----------|------|------|
| `.env` | ✅ .gitignore で除外 | API キーを格納。Git 管理外 |
| `.env.example` | ✅ 安全 | テンプレートのみ、実際の値なし |

### License Check

- `LICENSE` ファイル: **未設定**（要追加）

---

## 使い方

```bash
# インストール
pip install -e .

# 環境変数設定
cp .env.example .env
# .env に GEMINI_API_KEY を設定

# 米国政府ニュース収集
midas collect --source=us-gov
```

---

## 実装状況

### 実装済み ✅

- プロジェクト構造のセットアップ
- 米国政府ニュース収集エージェント
  - RSS フィード取得（5 ソース）
  - LLM による構造変化の判定・フィルタリング
  - JSON ファイルへの保存
- Claude Code カスタムコマンド（/analyze-project）

### 未実装（Phase 1）

- 株価データ収集
- 基本的な企業分析機能
- CLI でのレポート出力

### 未実装（Phase 2）

- その他ニュースソース（新技術ウォッチ等）
- 情報選別フィルタリングの強化
- PostgreSQL でのデータ永続化

### 未実装（Phase 3）

- 株価急変事例の収集・分析
- 知見のフィードバックループ
- 通知機能（Slack / LINE）
