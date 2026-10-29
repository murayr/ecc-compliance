"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { AnswerBubble, Icon, type Turn } from "./Message";

// Where the FastAPI backend runs. Baked in at build time (NEXT_PUBLIC_ variables are).
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Arabic first: Arabic suggestions lead. All four are questions from the eval set.
const SUGGESTIONS = [
  { lang: "ar", text: "متى يكون التحقق متعدد العناصر مطلوباً؟" },
  { lang: "ar", text: "هل يجب اختبار القدرة على استعادة النسخ الاحتياطية؟" },
  { lang: "en", text: "Can the cybersecurity department sit under the IT department?" },
  { lang: "en", text: "What must a vendor do with our data when their contract ends?" },
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const busy = turns.some((t) => t.status === "loading");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Keep the newest message in view.
  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    endRef.current?.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "end" });
  }, [turns]);

  // Grow the input with its content, up to a limit set in CSS.
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [draft]);

  async function ask(question: string, id = crypto.randomUUID()) {
    question = question.trim();
    if (question.length < 3 || busy) return;
    setTurns((ts) => {
      const exists = ts.some((t) => t.id === id);
      const turn: Turn = { id, question, status: "loading" };
      return exists ? ts.map((t) => (t.id === id ? turn : t)) : [...ts, turn];
    });
    setDraft("");
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : "تعذّر معالجة السؤال");
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, status: "done", result: body } : t)));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setTurns((ts) => ts.map((t) => (t.id === id ? { ...t, status: "error", error: message } : t)));
    } finally {
      inputRef.current?.focus();
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    ask(draft);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends, Shift+Enter adds a line. Never send mid-composition (input methods).
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      ask(draft);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark"><Icon name="shield" /></span>
          <div className="brand-text">
            <span className="brand-ar">مساعد الضوابط الأساسية للأمن السيبراني</span>
            <span className="brand-en" lang="en" dir="ltr">ECC Compliance Assistant</span>
          </div>
        </div>
        <span className="badge" lang="en" dir="ltr">ECC-2:2024</span>
      </header>

      <main className="thread" role="log" aria-live="polite" aria-label="المحادثة">
        {turns.length === 0 ? (
          <section className="welcome">
            <span className="welcome-mark"><Icon name="shield" /></span>
            <h1>كيف يمكنني مساعدتك في الضوابط الأساسية للأمن السيبراني؟</h1>
            <p className="welcome-en" lang="en" dir="ltr">
              Ask about the NCA Essential Cybersecurity Controls in Arabic or English.
            </p>
            <p className="welcome-note">
              كل إجابة مستندة إلى نص الضوابط ومرفقة برقم الضابط، وإن لم تغطِّ الضوابط سؤالك فسنخبرك بذلك.
            </p>
            <ul className="suggestions">
              {SUGGESTIONS.map((s) => (
                <li key={s.text}>
                  <button
                    type="button"
                    className="suggestion"
                    lang={s.lang}
                    dir={s.lang === "ar" ? "rtl" : "ltr"}
                    onClick={() => ask(s.text)}
                  >
                    <Icon name="question" />
                    <span className="suggestion-text">{s.text}</span>
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ) : (
          turns.map((turn) => (
            <article key={turn.id} className="turn">
              <div className="bubble user" dir="auto">{turn.question}</div>
              <AnswerBubble turn={turn} onRetry={() => ask(turn.question, turn.id)} />
            </article>
          ))
        )}
        <div ref={endRef} />
      </main>

      <footer className="composer-wrap">
        <form className="composer" onSubmit={onSubmit}>
          <textarea
            ref={inputRef}
            dir="auto"
            rows={1}
            value={draft}
            maxLength={1000}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="اكتب سؤالك بالعربية أو الإنجليزية…"
            aria-label="سؤالك"
            autoFocus
          />
          <button type="submit" className="send" disabled={busy || draft.trim().length < 3} aria-label="إرسال">
            <Icon name="send" />
          </button>
        </form>
        <p className="hint">
          <span>الإجابات مبنية على نص الضوابط فقط ولا تُغني عن المراجعة النظامية</span>
          {/* <bdi> isolates the English run so it can't reorder the Arabic around it */}
          <bdi className="hint-en" lang="en">Enter to send, Shift+Enter for a new line</bdi>
        </p>
      </footer>
    </div>
  );
}
