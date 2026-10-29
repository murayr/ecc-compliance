"use client";

import { useState } from "react";

type Control = {
  id: string;
  domain: string;
  subdomain_id: string;
  subdomain: string;
  text: string;
  score: number;
};

export type AskResponse = {
  lang: "en" | "ar";
  answer: string;
  refused: boolean;
  cited: Control[];
  model: string;
  usage: { input_tokens: number; output_tokens: number; cost_usd: number | null };
};

export type Turn = {
  id: string;
  question: string;
  status: "loading" | "done" | "error";
  result?: AskResponse;
  error?: string;
};

const LABELS = {
  ar: {
    searching: "جارٍ البحث في الضوابط…",
    cited: "الضوابط المستشهد بها",
    notCovered: "غير مغطى في الضوابط المتاحة",
    notCoveredHint: "لا تتضمن الضوابط ما يجيب عن هذا السؤال. جرّب صياغة أخرى أو اسأل عن موضوع آخر.",
    failed: "تعذّر الحصول على إجابة",
    retry: "إعادة المحاولة",
  },
  en: {
    searching: "Searching the controls…",
    cited: "Cited controls",
    notCovered: "Not covered by the available controls",
    notCoveredHint: "The controls don't answer this question. Try rephrasing, or ask about another topic.",
    failed: "Couldn't get an answer",
    retry: "Try again",
  },
};

const isArabic = (s: string) => /[؀-ۿ]/.test(s);
// A citation plus any punctuation right after it, so the two never wrap onto separate lines.
const CITATION = /(\[\d+(?:-\d+)+\][.,،؛:]?)/;

// Arabic subdomain titles carry the English name in brackets, e.g.
// "إدارة الأصول (Asset Management)". Split them so each script gets its own line and direction.
function splitTitle(title: string) {
  const m = title.match(/^(.*?)\s*\(\s*[”“"]?([^()]*?[A-Za-z][^()]*?)\s*\)\s*$/);
  return m && isArabic(m[1]) ? { main: m[1], english: m[2].replace(/[”“"]/g, "").trim() } : { main: title, english: null };
}

export function AnswerBubble({ turn, onRetry }: { turn: Turn; onRetry: () => void }) {
  const [openId, setOpenId] = useState<string | null>(null);
  const lang = turn.result?.lang ?? (isArabic(turn.question) ? "ar" : "en");
  const t = LABELS[lang];
  const dir = lang === "ar" ? "rtl" : "ltr";

  if (turn.status === "loading") {
    return (
      <div className="bubble assistant" lang={lang} dir={dir} aria-busy="true">
        <span className="avatar"><Icon name="shield" /></span>
        <div className="typing">
          <span className="dots" aria-hidden="true"><i /><i /><i /></span>
          <span>{t.searching}</span>
        </div>
      </div>
    );
  }

  if (turn.status === "error" || !turn.result) {
    return (
      <div className="bubble assistant error" lang={lang} dir={dir} role="alert">
        <span className="avatar"><Icon name="alert" /></span>
        <div className="body">
          <p className="title">{t.failed}</p>
          <p className="detail" dir="auto">{turn.error}</p>
          <button type="button" className="ghost" onClick={onRetry}>
            <Icon name="retry" /> {t.retry}
          </button>
        </div>
      </div>
    );
  }

  const r = turn.result;
  const openCard = (ref: string) => {
    const id = ref.slice(1, -1).split("-").slice(0, 3).join("-");    // [2-2-3-1] -> 2-2-3
    setOpenId(id);
    document.getElementById(`${turn.id}-${id}`)?.scrollIntoView({ block: "nearest" });
  };

  return (
    <div className={`bubble assistant${r.refused ? " refused" : ""}`} lang={lang} dir={dir}>
      <span className="avatar"><Icon name={r.refused ? "info" : "shield"} /></span>
      <div className="body">
        {r.refused ? (
          <>
            <p className="title">{t.notCovered}</p>
            <p className="detail">{t.notCoveredHint}</p>
          </>
        ) : (
          <p className="answer">
            {r.answer.split(CITATION).map((part, i) => {
              if (i % 2 === 0) return part;
              const ref = part.slice(0, part.lastIndexOf("]") + 1);
              return (
                <span key={i} className="ref-wrap">
                  <button type="button" className="ref" dir="ltr" onClick={() => openCard(ref)}>
                    {ref.slice(1, -1)}
                  </button>
                  {part.slice(ref.length)}
                </span>
              );
            })}
          </p>
        )}

        {r.cited.length > 0 && (
          <section className="sources" aria-label={t.cited}>
            <h2>{t.cited}</h2>
            {r.cited.map((c) => {
              const title = splitTitle(c.subdomain);
              return (
              <details
                key={c.id}
                id={`${turn.id}-${c.id}`}
                className="card"
                open={openId === c.id}
                onToggle={(e) => {
                  const open = (e.currentTarget as HTMLDetailsElement).open;
                  if (open) setOpenId(c.id);
                  else if (openId === c.id) setOpenId(null);
                }}
              >
                <summary>
                  <span className="card-id" dir="ltr">{c.id}</span>
                  <span className="card-title">
                    {title.main}
                    {title.english && <bdi className="card-title-en" lang="en">{title.english}</bdi>}
                  </span>
                  <span className="chevron" aria-hidden="true"><Icon name="chevron" /></span>
                </summary>
                <div className="card-body">
                  <p className="card-domain">{c.domain}</p>
                  <p className="card-text">{c.text}</p>
                </div>
              </details>
              );
            })}
          </section>
        )}

        <p className="meta" lang="en" dir="ltr">
          <span>{r.model}</span>
          <span>
            {r.usage.input_tokens.toLocaleString("en")} in, {r.usage.output_tokens.toLocaleString("en")} out tokens
          </span>
          {r.usage.cost_usd !== null && <span>~${r.usage.cost_usd.toFixed(4)}</span>}
        </p>
      </div>
    </div>
  );
}

const PATHS: Record<string, string> = {
  shield: "M12 3l7 3v5c0 4.6-3 8.6-7 10-4-1.4-7-5.4-7-10V6l7-3zm-3 9l2 2 4-4",
  send: "M4 12l16-8-6 16-2.5-6.5L4 12zm7.5 1.5L20 4",
  alert: "M12 8v5m0 3.5v.5M10.3 4.2L2.8 17.5A2 2 0 004.5 20.5h15a2 2 0 001.7-3L13.7 4.2a2 2 0 00-3.4 0z",
  info: "M12 21a9 9 0 100-18 9 9 0 000 18zm0-12.5v.5m0 3v5",
  retry: "M4 12a8 8 0 0113.7-5.6L20 8.7M20 4v4.7h-4.7M20 12a8 8 0 01-13.7 5.6L4 15.3M4 20v-4.7h4.7",
  chevron: "M6 9l6 6 6-6",
  question: "M12 21a9 9 0 100-18 9 9 0 000 18zm-2.5-11.5a2.5 2.5 0 115 0c0 1.7-2.5 2-2.5 3.5m0 3v.01",
};

export function Icon({ name }: { name: keyof typeof PATHS }) {
  return (
    <svg className={`icon icon-${name}`} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={PATHS[name]} />
    </svg>
  );
}
