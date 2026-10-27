import { useEffect, useState } from 'react';
import { DocumentUpload } from './DocumentUpload';

const modes = {
  rag: { label: 'RAG', title: 'Ground every answer in evidence.', description: 'A dedicated workspace for retrieval-augmented generation. PDF ingestion and local vector storage are available. Retrieval and cited answers are planned for a later milestone.', steps: ['Document library', 'Retrieval pipeline', 'Answers with citations'] },
  agentic: { label: 'Agentic', title: 'Make reasoning a visible process.', description: 'An independent execution mode for goal-driven workflows. Tools, orchestration and execution traces will arrive in a later milestone.', steps: ['Task workspace', 'Tool orchestration', 'Execution traces'] },
  compare: { label: 'Compare', title: 'Two approaches. One clear view.', description: 'A future workspace to compare RAG and Agentic runs against the same question. No evaluations or results are available yet.', steps: ['Shared test questions', 'Side-by-side results', 'Evaluation metrics'] },
} as const;
type Mode = keyof typeof modes;
function readMode(): Mode { const value = window.location.hash.slice(1); return Object.hasOwn(modes, value) ? value as Mode : 'rag'; }

export function App() {
  const [mode, setMode] = useState<Mode>(readMode);
  const [health, setHealth] = useState('Checking API...');
  useEffect(() => { const update = () => setMode(readMode()); window.addEventListener('hashchange', update); return () => window.removeEventListener('hashchange', update); }, []);
  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 5000);
    fetch('/api/health', { signal: controller.signal }).then(async response => {
      if (!response.ok) throw new Error('Health request failed');
      const body: unknown = await response.json();
      if (typeof body !== 'object' || body === null || !('status' in body) || body.status !== 'ok') throw new Error('Invalid health response');
      if (active) setHealth('API online');
    }).catch(() => { if (active) setHealth('API unavailable'); }).finally(() => window.clearTimeout(timer));
    return () => { active = false; controller.abort(); window.clearTimeout(timer); };
  }, []);
  const current = modes[mode];
  return <div className="app">
    <aside><a className="brand" href="#rag"><span className="mark">R<span> /</span></span> ResearchLab</a><p className="sidebar-label">AI ENGINEERING WORKSPACE</p>
      <nav aria-label="Execution modes">{Object.entries(modes).map(([key, item], index) => <a key={key} href={`#${key}`} aria-current={mode === key ? 'page' : undefined}><span className="nav-number">0{index + 1}</span>{item.label}<span className="nav-arrow" aria-hidden="true">&#8599;</span></a>)}</nav>
      <div className="sidebar-footer"><span className="tiny-dot"/> Foundation release <small>Milestone 2A / Local development</small></div>
    </aside>
    <div className="workspace"><header><span>Workspace <span className="slash">/</span> <strong>{current.label}</strong></span><span className="health" role="status">{health}</span></header>
      <main id="main"><div className="eyebrow">RESEARCH / BUILD / UNDERSTAND</div><div className="title-row"><h1>{current.label === 'RAG' ? 'Retrieval augmented generation' : current.label === 'Agentic' ? 'Agentic AI' : 'Compare modes'}</h1><span className="badge">{mode === 'rag' ? 'INGESTION READY' : 'PLANNED'}</span></div>
        <section className="hero" aria-labelledby="mode-title"><div className="hero-copy"><span className="section-number">EXPERIMENT {mode === 'rag' ? '01' : mode === 'agentic' ? '02' : '03'}</span><h2 id="mode-title">{current.title}</h2><p>{current.description}</p><div className="notice"><span aria-hidden="true">&#9719;</span> {mode === 'rag' ? 'Ingestion available / Question answering is not implemented' : 'Workspace preview / Execution is not implemented'}</div></div><div className="diagram" aria-hidden="true"><div className="orbit orbit-one"/><div className="orbit orbit-two"/><div className="core">{mode === 'rag' ? 'R' : mode === 'agentic' ? 'A' : '\u2194'}</div><span className="node node-one"/><span className="node node-two"/><span className="node node-three"/><span className="diagram-label">{mode.toUpperCase()} / RESEARCHLAB</span></div></section>
        {mode === 'rag' && <DocumentUpload />}
        <div className="section-heading"><h2>Workspace capabilities</h2><span>COMING IN LATER MILESTONES</span></div><div className="cards">{current.steps.map((step, index) => <article key={step}><span className="card-index">0{index + 1}</span><h3>{step}</h3><p>{mode === 'rag' && index === 0 ? 'PDF upload available above' : 'Not implemented'}</p></article>)}</div>
        <footer><span>Built for observable, testable AI engineering.</span><span>Foundation v0.1</span></footer>
      </main>
    </div>
  </div>;
}
