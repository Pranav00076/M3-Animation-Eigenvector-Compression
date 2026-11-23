import { useEffect, useState } from 'react';
import { Film, CheckCircle2, Play, Terminal, Layers } from 'lucide-react';

interface ReconstructionMetric {
  rank_k: number;
  stored_values: number;
  original_values: number;
  storage_ratio: number;
  space_saved_percent: number;
  frobenius_relative_error: number;
  rmse: number;
  psnr_db: number;
  energy_retained_percent: number;
  image_file: string | null;
}

interface PipelineMetrics {
  source_image: string;
  matrix_shape: [number, number];
  original_file_bytes: number;
  original_matrix_values: number;
  ata_symmetry_error: number;
  v_orthonormality_error: number;
  ata_reconstruction_rel_error: number;
  svd_reconstruction_rel_error: number;
  eigen_svd_lambda_sigma2_max_rel_diff: number;
  scipy_svd_max_rel_diff: number;
  reconstructions: ReconstructionMetric[];
}

const SCENES = [
  { id: 'full', label: 'Full Documentary (All 6 Scenes)', file: '/media/the_mathematics_behind_image_compression.mp4', span: 'Full Cut' },
  { id: 's1', label: 'Scene 1: The Mystery', file: '/media/scene01_intro.mp4', span: '0:00 – 0:35' },
  { id: 's2', label: 'Scene 2: Images as Matrices', file: '/media/scene02_pixels.mp4', span: '0:35 – 1:15' },
  { id: 's3', label: 'Scene 3: Eigenvectors & Diagonalization', file: '/media/scene03_eigenvectors.mp4', span: '1:15 – 2:05' },
  { id: 's4', label: 'Scene 4: Singular Value Decomposition', file: '/media/scene04_svd.mp4', span: '2:05 – 3:10' },
  { id: 's5', label: 'Scene 5: Low-Rank Reconstruction', file: '/media/scene05_reconstruction.mp4', span: '3:10 – 4:20' },
  { id: 's6', label: 'Scene 6: Conclusion', file: '/media/scene06_conclusion.mp4', span: '4:20 – 5:00' },
];

export default function App() {
  const [activeScene, setActiveScene] = useState(SCENES[0]);
  const [metrics, setMetrics] = useState<PipelineMetrics | null>(null);

  useEffect(() => {
    fetch('/media/computed_assets/metrics.json')
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setMetrics(data);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-[#050914] text-slate-100">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <header className="mb-8 flex flex-col gap-3 border-b border-cyan-500/20 pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-semibold tracking-[0.2em] text-cyan-400 uppercase">
              Manim Community Edition · NumPy · SciPy · Pillow
            </p>
            <h1 className="mt-1 text-3xl font-bold tracking-tight text-white">
              The Mathematics Behind Image Compression
            </h1>
            <p className="mt-1 text-sm text-slate-400">
              How Eigenvectors and Linear Algebra Help Reduce Image Data
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs text-cyan-300">
            <Film className="h-4 w-4" />
            <span>Rendered MP4 Screening Room</span>
          </div>
        </header>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <div className="overflow-hidden rounded-xl border border-cyan-500/30 bg-[#0B1326] shadow-2xl shadow-cyan-950/30">
              <video
                key={activeScene.file}
                src={activeScene.file}
                controls
                playsInline
                className="aspect-video w-full bg-black"
              />
              <div className="flex items-center justify-between border-t border-slate-800 px-4 py-3 text-sm">
                <div className="flex items-center gap-2 font-medium text-white">
                  <Play className="h-4 w-4 text-cyan-400" />
                  <span>{activeScene.label}</span>
                </div>
                <span className="font-mono text-xs text-slate-400">{activeScene.span}</span>
              </div>
            </div>

            <div className="mt-4 rounded-xl border border-slate-800 bg-[#0B1326]/80 p-4">
              <div className="mb-3 flex items-center gap-2 text-xs font-semibold tracking-wider text-slate-300 uppercase">
                <Layers className="h-4 w-4 text-blue-400" />
                <span>Scene Selector</span>
              </div>
              <div className="grid gap-2 sm:grid-cols-2">
                {SCENES.map((scene) => {
                  const selected = scene.id === activeScene.id;
                  return (
                    <button
                      key={scene.id}
                      type="button"
                      onClick={() => setActiveScene(scene)}
                      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-xs transition ${
                        selected
                          ? 'border-cyan-400 bg-cyan-500/15 text-white'
                          : 'border-slate-800 bg-[#050914]/60 text-slate-300 hover:border-slate-700 hover:text-white'
                      }`}
                    >
                      <span className="truncate pr-2 font-medium">{scene.label}</span>
                      <span className="shrink-0 font-mono text-[11px] text-slate-400">{scene.span}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <aside className="space-y-4">
            <div className="rounded-xl border border-slate-800 bg-[#0B1326]/90 p-4">
              <div className="mb-3 flex items-center gap-2 text-xs font-semibold tracking-wider text-purple-300 uppercase">
                <CheckCircle2 className="h-4 w-4 text-purple-400" />
                <span>Verified NumPy / SciPy Math</span>
              </div>
              {metrics ? (
                <div className="space-y-2.5 text-xs">
                  <div className="flex justify-between border-b border-slate-800/80 pb-1.5">
                    <span className="text-slate-400">Matrix A Shape</span>
                    <span className="font-mono text-cyan-300">
                      {metrics.matrix_shape[0]} × {metrics.matrix_shape[1]}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800/80 pb-1.5">
                    <span className="text-slate-400">max |λᵢ − σᵢ²| / max</span>
                    <span className="font-mono text-purple-300">
                      {metrics.eigen_svd_lambda_sigma2_max_rel_diff.toExponential(2)}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-slate-800/80 pb-1.5">
                    <span className="text-slate-400">‖VᵀV − I‖_F</span>
                    <span className="font-mono text-blue-300">
                      {metrics.v_orthonormality_error.toExponential(2)}
                    </span>
                  </div>
                  <div className="flex justify-between pb-1">
                    <span className="text-slate-400">‖A − UΣVᵀ‖ / ‖A‖</span>
                    <span className="font-mono text-cyan-300">
                      {metrics.svd_reconstruction_rel_error.toExponential(2)}
                    </span>
                  </div>

                  <div className="pt-2">
                    <div className="mb-2 text-[11px] font-semibold tracking-wider text-slate-400 uppercase">
                      Rank-k Truncations
                    </div>
                    <div className="space-y-1.5">
                      {metrics.reconstructions.map((r) => (
                        <div
                          key={r.rank_k}
                          className="flex items-center justify-between rounded bg-[#050914]/70 px-2.5 py-1.5 font-mono text-[11px]"
                        >
                          <span className="text-white">k={r.rank_k}</span>
                          <span className="text-cyan-300">{r.space_saved_percent.toFixed(1)}% saved</span>
                          <span className="text-purple-300">err {(r.frobenius_relative_error * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-400">Loading verified pipeline metrics...</p>
              )}
            </div>

            <div className="rounded-xl border border-slate-800 bg-[#0B1326]/90 p-4">
              <div className="mb-2 flex items-center gap-2 text-xs font-semibold tracking-wider text-cyan-300 uppercase">
                <Terminal className="h-4 w-4 text-cyan-400" />
                <span>1080p HD Render Command</span>
              </div>
              <pre className="overflow-x-auto rounded bg-[#050914] p-2.5 font-mono text-[11px] text-slate-300">
{`cd image-compression-animation
.venv/bin/python render_all.py --quality h`}
              </pre>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}
