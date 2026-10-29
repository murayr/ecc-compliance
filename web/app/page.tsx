"use client";

import { FormEvent, useState } from "react";

// Where the FastAPI backend runs. Baked in at build time (NEXT_PUBLIC_ variables are).
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Control = {
  id: string;
  domain: string;
  subdomain_id: string;
  subdomain: string;
  text: string;
  score: number;
};

type AskResponse = {
  lang: "en" | "ar";
  answer: string;
  refused: boolean;
  cited: Control[];
  model: string;
  usage: { input_tokens: number; output_tokens: number; cost_usd: number | null };
};

const LABELS = {
  en: { cited: "Cited controls", notCovered: "Not covered by the ECC controls" },
  ar: { cited: "الضوابط المستشهد بها", notCovered: "غير مغطى في الضوابط المتاحة" },
};

export default function Home() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AskResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const body = await res.json();
      if (!res.ok) {
        // FastAPI puts errors in "detail" (a string, or a list for validation errors)
        throw new Error(typeof body.detail === "string" ? body.detail : "Please enter a longer question.");
      }
      setResult(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const isArabic = result?.lang === "ar";
  const t = LABELS[result?.lang ?? "en"];

  return (
    <main>
      <h1>ECC Assistant</h1>
      <p className="subtitle">
        Ask about Saudi NCA's Essential Cybersecurity Controls (ECC-2:2024) in English or Arabic.
        <br />
        <span dir="rtl" lang="ar">اسأل عن الضوابط الأساسية للأمن السيبراني بالعربية أو الإنجليزية.</span>
      </p>

      <form onSubmit={onSubmit}>
        {/* dir="auto" flips the input to right-to-left as soon as you type Arabic */}
        <input
          dir="auto"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. When is multi-factor authentication required?"
          maxLength={1000}
          aria-label="Question"
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "…" : "Ask"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <section dir={isArabic ? "rtl" : "ltr"} lang={result.lang} className={isArabic ? "ar" : ""}>
          {result.refused ? (
            <div className="answer refused">{t.notCovered}</div>
          ) : (
            <div className="answer">
              {/* Keep cited IDs like [2-2-3] left-to-right inside Arabic text */}
              {result.answer.split(/(\[\d+(?:-\d+)+\])/).map((part, i) =>
                i % 2 ? <bdi key={i} dir="ltr" className="ref">{part}</bdi> : part
              )}
            </div>
          )}

          {result.cited.length > 0 && (
            <>
              <h2>{t.cited}</h2>
              {result.cited.map((c) => (
                <details key={c.id} className="card">
                  <summary>
                    <span className="id" dir="ltr">{c.id}</span>
                    <span>{c.subdomain}</span>
                  </summary>
                  <p className="domain">{c.domain}</p>
                  <p className="control-text">{c.text}</p>
                </details>
              ))}
            </>
          )}

          <p className="usage" dir="ltr">
            {result.model} · {result.usage.input_tokens} in / {result.usage.output_tokens} out tokens
            {result.usage.cost_usd !== null && ` · ~$${result.usage.cost_usd.toFixed(4)}`}
          </p>
        </section>
      )}
    </main>
  );
}
