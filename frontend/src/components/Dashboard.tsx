import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Mic, MicOff, Send, Volume2, VolumeX, RefreshCw, CheckCircle2,
  Info, Clock, Database, Layers,
  Search, Shield, Activity, Sparkles,
  Zap, Radio, Cpu, HardDrive, BarChart2
} from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Source {
  chunk_id: string;
  text: string;
  score: number;
  metadata: {
    document_id: string;
    strategy: string;
    language: string;
    parent_id?: string;
    is_parent?: boolean;
    source_query?: string;
  };
}

interface Latency {
  stt_ms?: number;
  embedding_ms?: number;
  dense_ms?: number;
  sparse_ms?: number;
  fusion_ms?: number;
  reranking_ms?: number;
  retrieval_total_ms?: number;
  generation_ms?: number;
  grounding_ms?: number;
  total_ms: number;
}

interface PipelineSteps {
  dense_candidates: number;
  sparse_candidates: number;
  fused_candidates: number;
  reranked_candidates: number;
}

interface GroundingDetails {
  retrieval_max_relevance: number;
  relevance_pass: boolean;
  semantic_similarity: number;
  word_intersection: number;
  llm_judge_verdict: number;
  llm_judge_reason: string;
  overall_score: number;
  verdict_reason: string;
}

interface QueryResponse {
  request_id: string;
  query: string;
  transcription?: string;
  answer: string;
  sources: Source[];
  language: string;
  grounded: boolean;
  confidence: number;
  status: string;
  latency: Latency;
  pipeline_steps: PipelineSteps;
  grounding_details?: GroundingDetails;
  cached?: boolean;
}

interface HealthStatus {
  status: string;
  environment: string;
  embedding_mode: string;
  reranker_type: string;
}

interface DatasetStatus {
  timestamp: number;
  environment: string;
  embedding_mode: string;
  reranker_type: string;
  elevenlabs_configured: boolean;
  groq_configured: boolean;
  qdrant: { status: string; vector_count: number; collection: string; is_mock: boolean };
  bm25: { status: string; doc_count: number; loaded: boolean };
  embedder: { status: string; model: string; dim: number; mode: string };
}

interface CumulativeMetrics {
  total_requests: number;
  cache_hits: number;
  cache_hit_rate: number;
  grounding_passes: number;
  grounding_rate: number;
  latency_p50_ms: number;
  latency_p70_ms: number;
  latency_p95_ms: number;
  latency_p100_ms: number;
}

// ─── Stage definitions for the live trace ────────────────────────────────────

type StageKey = 'embedding' | 'dense_retrieval' | 'sparse_retrieval' | 'fusion' | 'reranking' | 'generation_start' | 'generation' | 'grounding';
type PipelineState = 'idle' | 'loading' | 'streaming' | 'completed' | 'recording' | 'transcribing' | 'error';

interface StageState {
  latency_ms?: number;
  candidates?: number;
  done: boolean;
  grounded?: boolean;
  confidence?: number;
}

interface SSEEventPayload {
  type?: string;
  stage?: StageKey;
  latency_ms?: number;
  candidates?: number;
  grounded?: boolean;
  confidence?: number;
  text?: string;
  token?: string;
  answer?: string;
  message?: string;
  request_id?: string;
  query?: string;
  sources?: Source[];
  language?: string;
  status?: string;
  latency?: Partial<Latency>;
  pipeline_steps?: Partial<PipelineSteps>;
  grounding_details?: GroundingDetails;
  cached?: boolean;
}

// ─── Helper ──────────────────────────────────────────────────────────────────

function badge(color: string, text: string) {
  const colorMap: Record<string, [string, string]> = {
    green:  ['rgba(16,185,129,0.12)', '#34d399'],
    red:    ['rgba(239,68,68,0.12)',  '#f87171'],
    yellow: ['rgba(251,191,36,0.12)', '#fbbf24'],
    blue:   ['rgba(99,102,241,0.12)', '#818cf8'],
    gray:   ['rgba(71,85,105,0.15)',  '#94a3b8'],
  };
  const [bg, fg] = colorMap[color] || colorMap.gray;
  return (
    <span style={{ fontSize: '11px', background: bg, color: fg, border: `1px solid ${fg}30`, padding: '2px 8px', borderRadius: '4px', fontWeight: 600, letterSpacing: '0.04em' }}>
      {text}
    </span>
  );
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function formatNumber(value: unknown, digits: number, fallback = '0'): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : fallback;
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function Dashboard() {
  // ── Input ──
  const [query, setQuery] = useState('');
  const [language, setLanguage] = useState('hi');
  const [isRecording, setIsRecording] = useState(false);
  const [recordTime, setRecordTime] = useState(0);

  // ── Pipeline state ──
  const [pipelineState, setPipelineState] = useState<PipelineState>('idle');
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [isStreamingTokens, setIsStreamingTokens] = useState(false);
  const [stages, setStages] = useState<Partial<Record<StageKey, StageState>>>({});
  const [isCacheHit, setIsCacheHit] = useState(false);

  // ── API data ──
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [metrics, setMetrics] = useState<CumulativeMetrics | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);

  // ── Audio ──
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioChunksRef = useRef<Uint8Array[]>([]);

  // ── Refs ──
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioDataChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const sseAbortRef = useRef<AbortController | null>(null);
  const answerBoxRef = useRef<HTMLDivElement>(null);
  const [strategyFilter, setStrategyFilter] = useState<string>('all');

  // ── Mount: fetch health, status, metrics ──
  useEffect(() => {
    fetchHealth();
    fetchDatasetStatus();
    fetchMetrics();
    const interval = setInterval(() => { fetchMetrics(); fetchDatasetStatus(); }, 15000);
    return () => clearInterval(interval);
  }, []);

  // ── Recording timer ──
  useEffect(() => {
    if (isRecording) {
      timerRef.current = window.setInterval(() => setRecordTime(t => t + 1), 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecordTime(0);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [isRecording]);

  // ── Auto-scroll answer box ──
  useEffect(() => {
    if (answerBoxRef.current) answerBoxRef.current.scrollTop = answerBoxRef.current.scrollHeight;
  }, [streamingAnswer]);

  // ─── API calls ──────────────────────────────────────────────────────────────

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) setHealth(await res.json());
    } catch { /* silent */ }
  };

  const fetchDatasetStatus = async () => {
    setStatusLoading(true);
    try {
      const res = await fetch('/api/v1/status');
      if (res.ok) setDatasetStatus(await res.json());
    } catch { /* silent */ }
    setStatusLoading(false);
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/v1/metrics');
      if (res.ok) setMetrics(await res.json());
    } catch { /* silent */ }
  };

  // ─── SSE Text Streaming ─────────────────────────────────────────────────────

  const parseSSEBlock = (block: string): SSEEventPayload | null => {
    const data = block
      .split(/\r?\n/)
      .filter(line => line.startsWith('data:'))
      .map(line => line.replace(/^data:\s?/, ''))
      .join('\n')
      .trim();

    if (!data) return null;
    if (data === '[DONE]') return { type: 'done' };

    try {
      return JSON.parse(data);
    } catch (error) {
      console.warn('Unable to parse SSE payload:', data, error);
      return null;
    }
  };

  const submitTextQuerySSE = useCallback(async (queryText: string) => {
    // Cancel any ongoing SSE
    if (sseAbortRef.current) sseAbortRef.current.abort();
    sseAbortRef.current = new AbortController();

    setPipelineState('loading');
    setStreamingAnswer('');
    setIsStreamingTokens(false);
    setStages({});
    setResponse(null);
    setIsCacheHit(false);
    setAudioError(null);

    try {
      const res = await fetch('/api/v1/text/stream/sse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, language, latency_sensitive: false }),
        signal: sseAbortRef.current.signal,
      });

      if (!res.ok || !res.body) {
        setPipelineState('error');
        setStreamingAnswer(`Request failed with HTTP ${res.status}`);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let receivedEvent = false;

      while (true) {
        const { done, value } = await reader.read();
        buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
        const blocks = buffer.split(/\r?\n\r?\n/);
        buffer = blocks.pop() || '';

        for (const block of blocks) {
          const evt = parseSSEBlock(block);
          if (evt) {
            receivedEvent = true;
            handleSSEEvent(evt, queryText);
          }
        }

        if (done) break;
      }

      const trailingEvent = parseSSEBlock(buffer);
      if (trailingEvent) {
        receivedEvent = true;
        handleSSEEvent(trailingEvent, queryText);
      }

      setPipelineState(prev => {
        if (prev === 'loading' || prev === 'streaming') return receivedEvent ? 'completed' : 'error';
        return prev;
      });
      if (!receivedEvent) {
        setStreamingAnswer('The server closed the stream without sending any SSE events.');
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        console.error('SSE error:', e);
        setPipelineState('error');
        setStreamingAnswer(e?.message || 'SSE stream failed.');
      }
    }
  }, [language]);

  const handleSSEEvent = (evt: SSEEventPayload, submittedQuery: string = query) => {
    if (evt.type === 'stage') {
      setPipelineState(prev => prev === 'loading' ? 'streaming' : prev);
      if (!evt.stage) return;
      setStages(prev => ({ ...prev, [evt.stage as StageKey]: { latency_ms: evt.latency_ms ?? 0, candidates: evt.candidates, done: true, grounded: evt.grounded, confidence: evt.confidence } }));
    } else if (evt.type === 'cache_hit') {
      setPipelineState(prev => prev === 'loading' ? 'streaming' : prev);
      setIsCacheHit(true);
    } else if (evt.type === 'token') {
      const tokenText = evt.text ?? evt.token ?? evt.answer ?? '';
      if (!tokenText) return;
      setPipelineState('streaming');
      setIsStreamingTokens(true);
      setStreamingAnswer(prev => prev + tokenText);
    } else if (evt.type === 'done') {
      setIsStreamingTokens(false);
      setPipelineState('completed');
      const finalAnswer = evt.answer ?? '';
      // Build a QueryResponse from the done event
      const r: QueryResponse = {
        request_id: evt.request_id || 'sse',
        query: evt.query || submittedQuery,
        answer: finalAnswer,
        sources: evt.sources || [],
        language: evt.language || language,
        grounded: evt.grounded ?? false,
        confidence: evt.confidence ?? 0,
        status: evt.status || 'UNKNOWN',
        latency: typeof evt.latency === 'object' ? { total_ms: 0, ...evt.latency } as Latency : { total_ms: 0 },
        pipeline_steps: {
          dense_candidates: 0,
          sparse_candidates: 0,
          fused_candidates: 0,
          reranked_candidates: 0,
          ...evt.pipeline_steps,
        },
        grounding_details: evt.grounding_details,
        cached: evt.cached || false,
      };
      setResponse(r);
      if (finalAnswer) {
        setStreamingAnswer(finalAnswer);
      }
      fetchMetrics();
    } else if (evt.type === 'error') {
      setPipelineState('error');
      setIsStreamingTokens(false);
      setStreamingAnswer(evt.message || 'The stream returned an error.');
    }
  };

  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    submitTextQuerySSE(query);
  };

  // ─── Voice Recording ────────────────────────────────────────────────────────

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioDataChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioDataChunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioDataChunksRef.current, { type: 'audio/webm' });
        stream.getTracks().forEach(t => t.stop());
        await submitVoiceQueryWithAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setPipelineState('recording');
    } catch (err) {
      console.error('Mic error:', err);
      // Fallback: simulate a voice query via SSE text
      setIsRecording(true);
      setPipelineState('recording');
      setTimeout(() => {
        setIsRecording(false);
        const mockMap: Record<string, string> = {
          hi: 'भारत की राजधानी क्या है?',
          ta: 'இந்தியாவின் தலைநகரம் எது?',
          en: 'what is the capital of India?',
        };
        submitTextQuerySSE(mockMap[language] || mockMap.en);
      }, 2000);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setPipelineState('transcribing');
    }
  };

  // ─── Voice → audio stream playback ─────────────────────────────────────────

  const submitVoiceQueryWithAudio = async (audioBlob: Blob) => {
    setPipelineState('streaming');
    setStreamingAnswer('');
    setStages({});
    setResponse(null);
    setIsPlayingAudio(false);
    setAudioError(null);
    audioChunksRef.current = [];

    const formData = new FormData();
    formData.append('file', audioBlob, 'query.webm');
    formData.append('language', language);

    try {
      const res = await fetch('/api/v1/voice/stream', { method: 'POST', body: formData });
      if (!res.ok || !res.body) { setPipelineState('error'); return; }

      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) chunks.push(value);
      }

      // All audio received — play it
      const totalLength = chunks.reduce((s, c) => s + c.length, 0);
      const merged = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length; }

      if (merged.length < 500) {
        setIsPlayingAudio(false);
        if (!datasetStatus?.elevenlabs_configured) {
          setAudioError('Note: Real voice TTS playback requires a valid ElevenLabs API key in .env');
        }
        setPipelineState('completed');
        fetchMetrics();
        return;
      }

      const blob = new Blob([merged], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);

      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.onplay = () => setIsPlayingAudio(true);
        audioRef.current.onended = () => { setIsPlayingAudio(false); URL.revokeObjectURL(url); };
        audioRef.current.onerror = () => { setIsPlayingAudio(false); setAudioError('ElevenLabs API key required for audio stream.'); };
        try {
          await audioRef.current.play();
        } catch (err: any) {
          setIsPlayingAudio(false);
          if (err.name !== 'AbortError') {
            setAudioError('ElevenLabs API key required for spoken audio playback.');
          }
        }
      }

      setPipelineState('completed');
      fetchMetrics();
    } catch (e) {
      console.error('Voice stream error:', e);
      setPipelineState('error');
    }
  };

  // ─── Also allow TTS replay from text answer ─────────────────────────────────

  const playTextAnswerAsTTS = async () => {
    if (!streamingAnswer || streamingAnswer === 'NOT_SUPPORTED') return;
    setIsPlayingAudio(false);
    setAudioError(null);

    try {
      const res = await fetch('/api/v1/text/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, language }),
      });
      if (!res.ok || !res.body) { setAudioError('TTS stream failed'); return; }

      const reader = res.body.getReader();
      const chunks: Uint8Array[] = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (value) chunks.push(value);
      }

      const totalLength = chunks.reduce((s, c) => s + c.length, 0);
      const merged = new Uint8Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length; }

      if (merged.length < 500) {
        setIsPlayingAudio(false);
        setAudioError('ElevenLabs API key required for real TTS voice stream.');
        return;
      }

      const blob = new Blob([merged], { type: 'audio/mpeg' });
      const url = URL.createObjectURL(blob);
      if (audioRef.current) {
        audioRef.current.src = url;
        audioRef.current.onplay = () => setIsPlayingAudio(true);
        audioRef.current.onended = () => { setIsPlayingAudio(false); URL.revokeObjectURL(url); };
        audioRef.current.onerror = () => { setIsPlayingAudio(false); setAudioError('ElevenLabs API key required for audio stream.'); };
        try {
          await audioRef.current.play();
        } catch (err: any) {
          setIsPlayingAudio(false);
          if (err.name !== 'AbortError') {
            setAudioError('ElevenLabs API key required for spoken audio playback.');
          }
        }
      }
    } catch (e) {
      setAudioError(String(e));
    }
  };


  // ─── Filter sources ─────────────────────────────────────────────────────────

  const filteredSources = (response?.sources || []).filter(s => {
    if (strategyFilter === 'all') return true;
    if (strategyFilter === 'sentence') return s.metadata.strategy === 'sentence';
    if (strategyFilter === 'semantic') return s.metadata.strategy === 'semantic';
    if (strategyFilter === 'hierarchical') return s.metadata.strategy?.startsWith('hierarchical');
    return true;
  });

  const isBusy = pipelineState === 'loading' || pipelineState === 'streaming' || pipelineState === 'recording' || pipelineState === 'transcribing';
  const hasAnswer = pipelineState === 'completed' || pipelineState === 'error' || (pipelineState === 'streaming' && streamingAnswer.length > 0);

  // ─── Stage list for the trace panel ─────────────────────────────────────────

  const TRACE_STAGES: { key: StageKey; label: string; icon: React.ReactNode }[] = [
    { key: 'embedding', label: 'Query Embedding', icon: <Cpu size={13} /> },
    { key: 'dense_retrieval', label: 'Dense Search (Qdrant)', icon: <Database size={13} /> },
    { key: 'sparse_retrieval', label: 'Sparse Search (BM25)', icon: <Search size={13} /> },
    { key: 'fusion', label: 'RRF Hybrid Fusion', icon: <Layers size={13} /> },
    { key: 'reranking', label: 'Reranking', icon: <BarChart2 size={13} /> },
    { key: 'generation', label: 'LLM Generation (Groq)', icon: <Zap size={13} /> },
    { key: 'grounding', label: 'Grounding Verification', icon: <Shield size={13} /> },
  ];

  // ─── Render ──────────────────────────────────────────────────────────────────

  return (
    <div style={{ padding: '20px', maxWidth: '1600px', margin: '0 auto' }}>
      <audio ref={audioRef} style={{ display: 'none' }} />

      {/* ── HEADER ── */}
      <header className="glass-panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '14px 24px', borderRadius: '14px', marginBottom: '20px', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ background: 'linear-gradient(135deg, #4f46e5, #06b6d4)', borderRadius: '10px', padding: '8px', display: 'flex' }}>
            <Sparkles size={22} color="white" />
          </div>
          <div>
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 800 }}>
              <span className="text-gradient">Indic RAG Console</span>
            </h1>
            <p style={{ margin: 0, fontSize: '11px', color: '#64748b' }}>Multilingual Grounded Q&A · Sub-200ms Retrieval · ElevenLabs Voice</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', fontSize: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
          {[
            { label: 'Backend', val: health?.environment || 'checking…' },
            { label: 'Embedder', val: health?.embedding_mode || '—' },
            { label: 'Reranker', val: health?.reranker_type || '—' },
          ].map(({ label, val }) => (
            <div key={label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
              <span style={{ color: '#475569' }}>{label}</span>
              <span style={{ fontWeight: 600, textTransform: 'capitalize' }}>{val}</span>
            </div>
          ))}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '6px 12px', background: health ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)', borderRadius: '8px', border: `1px solid ${health ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'}` }}>
            <span className={`status-dot ${health ? 'ok' : 'error'}`} />
            <span style={{ fontWeight: 700, fontSize: '11px', letterSpacing: '0.06em' }}>{health ? 'ONLINE' : 'OFFLINE'}</span>
          </div>
        </div>
      </header>

      {/* ── DATASET STATUS ROW ── */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '20px' }}>
        {/* Qdrant status */}
        <StatusCard
          icon={<HardDrive size={18} />}
          title="Qdrant Vector DB"
          loading={statusLoading}
          status={datasetStatus?.qdrant.is_mock ? 'warn' : (datasetStatus?.qdrant.status === 'connected' ? 'ok' : 'error')}
          primary={datasetStatus?.qdrant.is_mock ? 'MOCK' : (datasetStatus?.qdrant.status === 'connected' ? 'Connected' : (datasetStatus?.qdrant.status || 'Unknown'))}
          secondary={datasetStatus?.qdrant ? `${datasetStatus.qdrant.vector_count.toLocaleString()} vectors · ${datasetStatus.qdrant.collection}` : 'Checking…'}
          color="#818cf8"
        />
        {/* BM25 status */}
        <StatusCard
          icon={<Search size={18} />}
          title="BM25 Sparse Index"
          loading={statusLoading}
          status={datasetStatus?.bm25.loaded ? 'ok' : (datasetStatus?.bm25.status === 'empty' ? 'warn' : 'error')}
          primary={datasetStatus?.bm25.status === 'loaded' ? 'Loaded' : (datasetStatus?.bm25.status || 'Unknown')}
          secondary={datasetStatus?.bm25 ? `${datasetStatus.bm25.doc_count.toLocaleString()} chunks` : 'Checking…'}
          color="#34d399"
        />
        {/* Embedder status */}
        <StatusCard
          icon={<Cpu size={18} />}
          title="Embedding Model"
          loading={statusLoading}
          status={datasetStatus?.embedder.status === 'loaded' ? 'ok' : 'warn'}
          primary={datasetStatus?.embedder.mode || '—'}
          secondary={datasetStatus?.embedder ? `dim=${datasetStatus.embedder.dim} · paraphrase-MiniLM-L12` : 'Checking…'}
          color="#fbbf24"
        />
        {/* API status */}
        <StatusCard
          icon={<Radio size={18} />}
          title="API Keys"
          loading={statusLoading}
          status={(datasetStatus?.groq_configured && datasetStatus?.elevenlabs_configured) ? 'ok' : 'warn'}
          primary={datasetStatus ? (datasetStatus.groq_configured ? 'Groq ✓' : 'Groq ✗') + ' · ' + (datasetStatus.elevenlabs_configured ? 'ElevenLabs ✓' : 'ElevenLabs ✗') : 'Checking…'}
          secondary="Groq LLM · ElevenLabs STT+TTS"
          color="#f472b6"
        />

        {/* Cumulative metrics quick-stats */}
        <div className="glass-panel" style={{ padding: '14px 16px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Activity style={{ color: '#3b82f6', flexShrink: 0 }} size={20} />
          <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>Total Queries</div>
            <div style={{ fontSize: '22px', fontWeight: 800, lineHeight: 1.1 }}>{metrics?.total_requests ?? 0}</div>
            <div style={{ fontSize: '10px', color: '#475569', marginTop: '2px' }}>Grounding: {((metrics?.grounding_rate ?? 0) * 100).toFixed(0)}% · Cache: {((metrics?.cache_hit_rate ?? 0) * 100).toFixed(0)}%</div>
          </div>
        </div>
        <div className="glass-panel" style={{ padding: '14px 16px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <Clock style={{ color: '#a855f7', flexShrink: 0 }} size={20} />
          <div>
            <div style={{ fontSize: '11px', color: '#64748b' }}>Latency (P50/P95/P100)</div>
            <div style={{ fontSize: '18px', fontWeight: 800, lineHeight: 1.1 }}>
              {(metrics?.latency_p50_ms ?? 0).toFixed(0)} / {(metrics?.latency_p95_ms ?? 0).toFixed(0)} / {(metrics?.latency_p100_ms ?? 0).toFixed(0)} ms
            </div>
            <div style={{ fontSize: '10px', color: (metrics?.latency_p95_ms ?? 0) < 200 ? '#10b981' : '#f59e0b', marginTop: '2px' }}>
              {(metrics?.latency_p95_ms ?? 0) < 200 ? '✓ P95 < 200ms target' : '⚠ P95 > 200ms target'}
            </div>
          </div>
        </div>
      </section>

      {/* ── MAIN TWO-COLUMN GRID ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.3fr) minmax(0, 1fr)', gap: '20px', alignItems: 'start' }}>

        {/* ── LEFT COLUMN ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {/* Query Input Panel */}
          <div className="glass-panel" style={{ padding: '24px', borderRadius: '16px' }}>
            <h2 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Search size={18} style={{ color: '#818cf8' }} /> Query Input
            </h2>

            <form onSubmit={handleTextSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-end' }}>
                {/* Language */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <label style={{ fontSize: '10px', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Language</label>
                  <select
                    id="language-select"
                    value={language}
                    onChange={e => setLanguage(e.target.value)}
                    style={{ background: '#0f172a', border: '1px solid #1e3a5f', color: '#f8fafc', borderRadius: '8px', padding: '8px 12px', outline: 'none', height: '40px', fontSize: '13px' }}
                  >
                    <option value="hi">हिंदी (Hindi)</option>
                    <option value="ta">தமிழ் (Tamil)</option>
                    <option value="te">తెలుగు (Telugu)</option>
                    <option value="kn">ಕನ್ನಡ (Kannada)</option>
                    <option value="ml">മലയാളം (Malayalam)</option>
                    <option value="mr">मराठी (Marathi)</option>
                    <option value="gu">ગુજરાતી (Gujarati)</option>
                    <option value="bn">বাংলা (Bengali)</option>
                    <option value="pa">ਪੰਜਾਬੀ (Punjabi)</option>
                    <option value="or">ଓଡ଼ିଆ (Odia)</option>
                    <option value="as">অসমীয়া (Assamese)</option>
                    <option value="en">English</option>
                  </select>
                </div>

                {/* Text input */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <label style={{ fontSize: '10px', color: '#64748b', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Question</label>
                  <div style={{ position: 'relative', display: 'flex' }}>
                    <input
                      id="query-input"
                      type="text"
                      placeholder="Ask the knowledge base…"
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      disabled={isBusy}
                      style={{ flex: 1, background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: '8px', padding: '8px 48px 8px 16px', outline: 'none', height: '40px', color: '#f8fafc', fontSize: '14px', opacity: isBusy ? 0.6 : 1 }}
                    />
                    <button
                      id="submit-btn"
                      type="submit"
                      disabled={isBusy || !query.trim()}
                      style={{ position: 'absolute', right: '4px', top: '4px', background: isBusy ? '#334155' : 'linear-gradient(135deg, #4f46e5, #7c3aed)', color: 'white', border: 'none', borderRadius: '6px', width: '32px', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: isBusy ? 'not-allowed' : 'pointer', transition: 'all 0.2s' }}
                    >
                      <Send size={14} />
                    </button>
                  </div>
                </div>
              </div>

              {/* Voice bar */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(15, 23, 42, 0.6)', padding: '12px 16px', borderRadius: '10px', border: '1px solid #1e3a5f' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  <button
                    id="mic-btn"
                    type="button"
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={pipelineState === 'loading' || pipelineState === 'streaming' || pipelineState === 'transcribing'}
                    className={isRecording ? 'pulse-record' : ''}
                    style={{ background: isRecording ? '#dc2626' : 'linear-gradient(135deg, #1e293b, #0f172a)', border: isRecording ? 'none' : '1px solid #334155', width: '40px', height: '40px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', color: 'white', transition: 'all 0.2s' }}
                  >
                    {isRecording ? <MicOff size={16} /> : <Mic size={16} />}
                  </button>
                  <div>
                    <div style={{ fontSize: '13px', fontWeight: 600 }}>
                      {isRecording ? `🔴 Recording… ${recordTime}s` : 'Voice Input (ElevenLabs Scribe)'}
                    </div>
                    <div style={{ fontSize: '11px', color: '#475569' }}>
                      {isRecording ? 'Click stop to transcribe & stream answer' : 'Hold to record · streams TTS audio response'}
                    </div>
                  </div>
                </div>

                {/* Waveform during audio playback */}
                {isPlayingAudio ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '3px', height: '32px' }}>
                    {[...Array(7)].map((_, i) => <div key={i} className="wave-bar" />)}
                  </div>
                ) : pipelineState !== 'idle' && pipelineState !== 'completed' ? (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: '#818cf8' }}>
                    <RefreshCw size={13} className="spin" />
                    <span style={{ textTransform: 'uppercase', fontWeight: 700, letterSpacing: '0.06em' }}>{pipelineState}</span>
                  </div>
                ) : null}

                {/* TTS replay button */}
                {pipelineState === 'completed' && streamingAnswer && streamingAnswer !== 'NOT_SUPPORTED' && (
                  <button
                    id="tts-replay-btn"
                    type="button"
                    onClick={playTextAnswerAsTTS}
                    disabled={isPlayingAudio}
                    style={{ display: 'flex', alignItems: 'center', gap: '6px', background: isPlayingAudio ? '#334155' : 'rgba(129,140,248,0.12)', border: '1px solid rgba(129,140,248,0.3)', color: '#818cf8', borderRadius: '8px', padding: '6px 12px', fontSize: '12px', fontWeight: 600, cursor: isPlayingAudio ? 'not-allowed' : 'pointer' }}
                  >
                    {isPlayingAudio ? <VolumeX size={14} /> : <Volume2 size={14} />}
                    {isPlayingAudio ? 'Playing…' : 'Play TTS'}
                  </button>
                )}
              </div>

              {audioError && (
                <div style={{ fontSize: '12px', color: '#f87171', background: 'rgba(239,68,68,0.08)', padding: '8px 12px', borderRadius: '6px', border: '1px solid rgba(239,68,68,0.15)' }}>
                  ⚠ {audioError}
                </div>
              )}
            </form>
          </div>

          {/* Answer Panel */}
          {hasAnswer && (
            <div className="glass-panel slide-in" style={{ padding: '24px', borderRadius: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', gap: '8px', flexWrap: 'wrap' }}>
                <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <Sparkles size={16} style={{ color: '#fbbf24' }} /> Generated Answer
                </h2>
                <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                  {isCacheHit && badge('yellow', '⚡ CACHE HIT')}
                  {isStreamingTokens && badge('blue', '● STREAMING')}
                  {pipelineState === 'completed' && response && (
                    response.answer === 'NOT_SUPPORTED' || response.status === 'OUT_OF_SCOPE'
                      ? badge('red', 'OUT OF SCOPE')
                      : response.grounded
                        ? badge('green', '✓ GROUNDED')
                        : badge('red', '✗ HALLUCINATION')
                  )}
                  {pipelineState === 'completed' && response && (
                    badge('gray', `${response.confidence ? (response.confidence * 100).toFixed(0) : 0}% conf`)
                  )}
                </div>
              </div>

              {/* Transcription (voice) */}
              {response?.transcription && (
                <div style={{ padding: '8px 14px', background: 'rgba(129,140,248,0.08)', borderLeft: '3px solid #818cf8', borderRadius: '4px', marginBottom: '12px', fontSize: '13px' }}>
                  <span style={{ fontWeight: 600, color: '#818cf8', marginRight: '6px' }}>Transcript:</span>
                  "{response.transcription}"
                </div>
              )}

              {/* Streaming answer box */}
              <div
                ref={answerBoxRef}
                style={{ background: 'rgba(2, 6, 23, 0.6)', padding: '16px 20px', borderRadius: '10px', fontSize: '15px', lineHeight: '1.7', border: '1px solid rgba(99,102,241,0.15)', whiteSpace: 'pre-wrap', color: '#e2e8f0', minHeight: '80px', maxHeight: '280px', overflowY: 'auto' }}
              >
                {streamingAnswer === 'NOT_SUPPORTED' || streamingAnswer === 'blocked_by_safety' ? (
                  <span style={{ color: '#f87171', fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Info size={16} />
                    {streamingAnswer === 'blocked_by_safety'
                      ? 'Query blocked by safety guardrail.'
                      : 'Sufficient evidence was not found in the retrieved documents to answer this question.'}
                  </span>
                ) : (
                  <span className={isStreamingTokens ? 'streaming-cursor' : ''}>{streamingAnswer}</span>
                )}
              </div>

              {pipelineState === 'completed' && response && (
                <div style={{ marginTop: '10px', fontSize: '10px', color: '#475569', display: 'flex', justifyContent: 'space-between', fontFamily: 'JetBrains Mono, monospace' }}>
                  <span>req: {response.request_id}</span>
                  <span>E2E: {formatNumber(response.latency?.total_ms, 1)}ms</span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* ── RIGHT COLUMN: Pipeline Trace ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div className="glass-panel" style={{ padding: '24px', borderRadius: '16px' }}>
            <h2 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={18} style={{ color: '#fbbf24' }} /> Live Pipeline Trace
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {TRACE_STAGES.map(({ key, label, icon }) => {
                const stage = stages[key];
                const isActive = (pipelineState === 'loading' || pipelineState === 'streaming') && !stage;
                const isDone = !!stage?.done;

                return (
                  <div
                    key={key}
                    style={{ display: 'flex', flexDirection: 'column', gap: '4px', padding: '10px 12px', borderRadius: '8px', background: isDone ? 'rgba(16,185,129,0.06)' : isActive ? 'rgba(99,102,241,0.06)' : 'rgba(15,23,42,0.4)', border: `1px solid ${isDone ? 'rgba(16,185,129,0.15)' : isActive ? 'rgba(99,102,241,0.15)' : 'rgba(255,255,255,0.04)'}`, transition: 'all 0.3s ease' }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <span style={{ color: isDone ? '#34d399' : isActive ? '#818cf8' : '#475569' }}>
                          {isDone ? <CheckCircle2 size={13} /> : isActive ? <RefreshCw size={13} className="spin" /> : icon}
                        </span>
                        <span style={{ fontWeight: isDone ? 600 : 400, color: isDone ? '#e2e8f0' : '#94a3b8' }}>{label}</span>
                        {stage?.candidates !== undefined && (
                          <span style={{ fontSize: '10px', background: 'rgba(99,102,241,0.1)', color: '#818cf8', padding: '1px 6px', borderRadius: '3px' }}>
                            {stage.candidates} chunks
                          </span>
                        )}
                      </div>
                      <span style={{ color: isDone ? '#a855f7' : '#475569', fontWeight: 700, fontFamily: 'monospace', fontSize: '11px' }}>
                        {isDone ? `${formatNumber(stage?.latency_ms, 1)} ms` : '—'}
                      </span>
                    </div>

                    {/* Progress bar */}
                    {isDone && response && (
                      <div style={{ height: '3px', background: 'rgba(15,23,42,0.8)', borderRadius: '2px', overflow: 'hidden' }}>
                        <div style={{
                          height: '100%',
                          width: `${Math.min(100, ((numberOrZero(stage?.latency_ms) / (numberOrZero(response.latency?.total_ms) || 1)) * 100))}%`,
                          background: 'linear-gradient(90deg, #a855f7, #6366f1)',
                          borderRadius: '2px',
                          transition: 'width 0.4s ease'
                        }} />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Total latency */}
            {pipelineState === 'completed' && response && (
              <div style={{ marginTop: '14px', paddingTop: '14px', borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'space-between', fontSize: '14px', fontWeight: 800 }}>
                <span>Total E2E Latency</span>
                <span style={{ color: (response.latency?.total_ms || 0) < 200 ? '#10b981' : '#f59e0b' }}>
                  {formatNumber(response.latency?.total_ms, 1)} ms
                  <span style={{ fontSize: '10px', fontWeight: 400, color: '#64748b', marginLeft: '6px' }}>
                    {(response.latency?.total_ms || 0) < 200 ? '✓ < 200ms' : '⚠ > 200ms'}
                  </span>
                </span>
              </div>
            )}

            {/* Retrieval funnel */}
            {pipelineState === 'completed' && response?.pipeline_steps && (
              <div style={{ marginTop: '20px' }}>
                <h3 style={{ margin: '0 0 10px 0', fontSize: '12px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Retrieval Funnel</h3>
                {[
                  { label: 'Dense (Qdrant)', val: numberOrZero(response.pipeline_steps.dense_candidates), color: '#818cf8' },
                  { label: 'Sparse (BM25)', val: numberOrZero(response.pipeline_steps.sparse_candidates), color: '#34d399' },
                  { label: 'Fused (RRF)', val: numberOrZero(response.pipeline_steps.fused_candidates), color: '#fbbf24' },
                  { label: 'Final (reranked)', val: numberOrZero(response.pipeline_steps.reranked_candidates), color: '#f472b6' },
                ].map(step => {
                  const max = Math.max(numberOrZero(response.pipeline_steps.dense_candidates), numberOrZero(response.pipeline_steps.sparse_candidates), 1);
                  return (
                    <div key={step.label} style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '11px', marginBottom: '6px' }}>
                      <div style={{ width: '90px', color: '#64748b', flexShrink: 0 }}>{step.label}</div>
                      <div style={{ flex: 1, height: '12px', background: 'rgba(15,23,42,0.8)', borderRadius: '3px', overflow: 'hidden', position: 'relative' }}>
                        <div style={{ height: '100%', width: `${(step.val / max) * 100}%`, background: step.color, borderRadius: '3px', opacity: 0.8, transition: 'width 0.5s ease' }} />
                      </div>
                      <span style={{ color: step.color, fontWeight: 700, width: '28px', textAlign: 'right' }}>{step.val}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Grounding Auditor */}
          {pipelineState === 'completed' && response?.grounding_details && (
            <div className="glass-panel slide-in" style={{ padding: '24px', borderRadius: '16px' }}>
              <h2 style={{ margin: '0 0 16px 0', fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Shield size={18} style={{ color: '#10b981' }} /> Grounding Auditor
              </h2>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {[
                  { name: 'Retrieval Relevance', val: numberOrZero(response.grounding_details.retrieval_max_relevance), threshold: 0.35, desc: 'Max similarity from index' },
                  { name: 'Semantic Alignment', val: numberOrZero(response.grounding_details.semantic_similarity), threshold: 0.60, desc: 'Cosine sim: answer ↔ context' },
                  { name: 'Citation Coverage', val: numberOrZero(response.grounding_details.word_intersection), threshold: 0.40, desc: 'Word-level overlap ratio' },
                ].map(sig => {
                  const pass = sig.val >= sig.threshold;
                  return (
                    <div key={sig.name} style={{ background: 'rgba(2,6,23,0.5)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '8px', padding: '10px 12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, marginBottom: '2px' }}>
                        <span>{sig.name}</span>
                        <span style={{ color: pass ? '#34d399' : '#f87171' }}>{formatNumber(sig.val, 2)} ≥{sig.threshold}</span>
                      </div>
                      <div style={{ fontSize: '10px', color: '#475569', marginBottom: '6px' }}>{sig.desc}</div>
                      <div style={{ height: '5px', background: '#0f172a', borderRadius: '3px', overflow: 'hidden', position: 'relative' }}>
                        <div style={{ height: '100%', width: `${sig.val * 100}%`, background: pass ? '#10b981' : '#ef4444', borderRadius: '3px', transition: 'width 0.5s ease' }} />
                        <div style={{ position: 'absolute', left: `${sig.threshold * 100}%`, top: 0, bottom: 0, width: '2px', background: '#fbbf24' }} />
                      </div>
                    </div>
                  );
                })}
                <div style={{ padding: '10px 12px', background: 'rgba(99,102,241,0.06)', border: '1px solid rgba(99,102,241,0.12)', borderRadius: '8px', fontSize: '11px' }}>
                  <div style={{ fontWeight: 600, color: '#818cf8', marginBottom: '4px' }}>LLM Judge Verdict</div>
                  <div style={{ color: '#94a3b8', display: 'flex', gap: '6px' }}>
                    <Info size={13} style={{ flexShrink: 0, marginTop: '1px' }} />
                    <span>{response.grounding_details.llm_judge_reason || response.grounding_details.verdict_reason}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── BOTTOM: Retrieved Chunks ── */}
      {pipelineState === 'completed' && response && response.sources.length > 0 && (
        <section className="glass-panel slide-in" style={{ padding: '24px', borderRadius: '16px', marginTop: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: '16px' }}>
            <div>
              <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={18} style={{ color: '#fbbf24' }} /> Retrieved Chunks ({filteredSources.length})
              </h2>
              <p style={{ margin: '4px 0 0 0', fontSize: '11px', color: '#475569' }}>Indexed paragraphs from MSMARCO-XI dataset</p>
            </div>
            {/* Strategy filter */}
            <div style={{ display: 'flex', background: 'rgba(15,23,42,0.6)', padding: '3px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)' }}>
              {['all', 'sentence', 'semantic', 'hierarchical'].map(f => (
                <button
                  key={f}
                  onClick={() => setStrategyFilter(f)}
                  style={{ background: strategyFilter === f ? 'rgba(99,102,241,0.2)' : 'transparent', border: 'none', color: strategyFilter === f ? '#818cf8' : '#64748b', padding: '5px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '11px', fontWeight: strategyFilter === f ? 700 : 400, transition: 'all 0.15s' }}
                >
                  {f === 'all' ? 'All' : f}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '14px' }}>
            {filteredSources.map((source, index) => {
              const stratColor = source.metadata.strategy === 'semantic' ? '#10b981' : source.metadata.strategy?.startsWith('hierarchical') ? '#a855f7' : '#3b82f6';
              return (
                <div
                  key={source.chunk_id}
                  className="glass-card"
                  style={{ padding: '14px 16px', borderRadius: '10px', display: 'flex', flexDirection: 'column', gap: '10px', borderLeft: `3px solid ${stratColor}` }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '11px', background: 'rgba(99,102,241,0.1)', color: '#818cf8', padding: '2px 8px', borderRadius: '4px', fontWeight: 700, fontFamily: 'monospace' }}>
                      [{index + 1}] {source.chunk_id.replace(/^msmarco_/, '')}
                    </span>
                    <span style={{ fontSize: '12px', fontWeight: 700, color: numberOrZero(source.score) > 0.7 ? '#10b981' : numberOrZero(source.score) > 0.4 ? '#fbbf24' : '#f87171' }}>
                      {formatNumber(source.score, 3)}
                    </span>
                  </div>
                  <div style={{ fontSize: '13px', lineHeight: '1.55', color: '#cbd5e1', flex: 1 }}>
                    "{source.text}"
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', fontSize: '10px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                    <span style={{ background: 'rgba(59,130,246,0.1)', color: '#60a5fa', padding: '2px 6px', borderRadius: '3px' }}>
                      doc:{source.metadata.document_id}
                    </span>
                    <span style={{ background: `${stratColor}18`, color: stratColor, padding: '2px 6px', borderRadius: '3px' }}>
                      {source.metadata.strategy}
                    </span>
                    <span style={{ background: 'rgba(251,191,36,0.1)', color: '#fbbf24', padding: '2px 6px', borderRadius: '3px', textTransform: 'uppercase' }}>
                      {source.metadata.language}
                    </span>
                    {source.metadata.parent_id && (
                      <span style={{ background: 'rgba(168,85,247,0.1)', color: '#c084fc', padding: '2px 6px', borderRadius: '3px' }}>
                        parent:{source.metadata.parent_id}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

// ─── Status Card Sub-component ────────────────────────────────────────────────

function StatusCard({ icon, title, loading, status, primary, secondary, color }: {
  icon: React.ReactNode;
  title: string;
  loading: boolean;
  status: 'ok' | 'warn' | 'error' | 'idle';
  primary: string;
  secondary: string;
  color: string;
}) {
  return (
    <div className="glass-panel" style={{ padding: '14px 16px', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
      <div style={{ background: `${color}18`, borderRadius: '8px', padding: '8px', color, display: 'flex', flexShrink: 0 }}>
        {icon}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontSize: '10px', color: '#64748b', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '2px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span className={`status-dot ${status}`} />
          {title}
        </div>
        {loading ? (
          <>
            <div className="skeleton" style={{ height: '16px', width: '70%', marginBottom: '4px' }} />
            <div className="skeleton" style={{ height: '11px', width: '90%' }} />
          </>
        ) : (
          <>
            <div style={{ fontSize: '14px', fontWeight: 700, color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{primary}</div>
            <div style={{ fontSize: '10px', color: '#475569', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{secondary}</div>
          </>
        )}
      </div>
    </div>
  );
}


