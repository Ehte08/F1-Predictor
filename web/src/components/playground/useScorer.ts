"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  PlaygroundInput,
  PlaygroundResultRow,
  WorkerResponse,
} from "@/lib/types";

export type ScorerStatus = "idle" | "loading" | "ready" | "scoring" | "error";

export function useScorer(active: boolean) {
  const workerRef = useRef<Worker | null>(null);
  const [status, setStatus] = useState<ScorerStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [rows, setRows] = useState<PlaygroundResultRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<PlaygroundInput | null>(null);

  // Lazily spin up the worker (and the 16 MB model fetch) only once activated.
  useEffect(() => {
    if (!active || workerRef.current) return;
    const worker = new Worker(new URL("../../workers/scorer.worker.ts", import.meta.url));
    workerRef.current = worker;
    setStatus("loading");
    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const msg = e.data;
      if (msg.type === "progress") {
        setProgress(msg.total ? msg.loaded / msg.total : 0);
      } else if (msg.type === "ready") {
        setStatus("ready");
        setProgress(1);
        if (pending.current) {
          worker.postMessage({ type: "score", input: pending.current });
          pending.current = null;
          setStatus("scoring");
        }
      } else if (msg.type === "result") {
        setRows(msg.rows);
        setStatus("ready");
      } else if (msg.type === "error") {
        setError(msg.message);
        setStatus("error");
      }
    };
    worker.postMessage({ type: "init" });
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, [active]);

  const run = useCallback((input: PlaygroundInput) => {
    const worker = workerRef.current;
    if (!worker) {
      pending.current = input;
      return;
    }
    if (status === "loading") {
      pending.current = input;
      return;
    }
    setStatus("scoring");
    worker.postMessage({ type: "score", input });
  }, [status]);

  return { status, progress, rows, error, run };
}
