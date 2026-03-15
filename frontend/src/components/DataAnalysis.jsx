import { useState } from 'react'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, ReferenceLine } from 'recharts'
import { Activity, AlertTriangle, TrendingUp, BarChart2, Layers, Play } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899']

export default function DataAnalysis({ sessionId, onComplete, existingData }) {
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)
    const [activeSection, setActiveSection] = useState('stats')

    const handleAnalyze = async () => {
        setLoading(true)
        setError(null)
        try {
            const res = await axios.post(`${API_BASE}/api/analyze/${sessionId}`)
            onComplete(res.data)
        } catch (err) {
            setError(err.response?.data?.detail || 'Analysis failed')
        } finally {
            setLoading(false)
        }
    }

    const data = existingData

    const sections = [
        { id: 'stats', label: 'Statistics', icon: Activity },
        { id: 'domain', label: 'Domain Check', icon: AlertTriangle },
        { id: 'correlation', label: 'Correlations', icon: TrendingUp },
        { id: 'univariate', label: 'Univariate', icon: BarChart2 },
        { id: 'bivariate', label: 'Bivariate', icon: Layers },
    ]

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-2xl font-bold text-slate-900">Data Analysis</h1>
                <p className="mt-1 text-slate-500">Deep statistical analysis, quality checks, correlations, and feature distributions.</p>
            </div>

            {!data ? (
                <div className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {[
                            { icon: Activity, title: 'Statistical Analysis', desc: 'Mean, std, min, max, median, skewness, kurtosis per column.' },
                            { icon: AlertTriangle, title: 'Domain Quality Check', desc: 'Detects corrupt values, outliers, zero-variance columns, and high missing %.' },
                            { icon: TrendingUp, title: 'Correlation Analysis', desc: 'Feature-target correlations + multicollinearity detection between features.' },
                            { icon: BarChart2, title: 'Univariate & Bivariate', desc: 'Distributions, histograms, and feature-vs-target scatter analysis.' },
                        ].map(({ icon: Icon, title, desc }) => (
                            <div key={title} className="bg-white border border-slate-200 rounded-xl p-4 flex gap-3">
                                <Icon className="h-5 w-5 text-indigo-500 shrink-0 mt-0.5" />
                                <div>
                                    <p className="font-semibold text-slate-800 text-sm">{title}</p>
                                    <p className="text-slate-500 text-xs mt-1">{desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                    {error && <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">{error}</div>}
                    <button onClick={handleAnalyze} disabled={loading}
                        className="w-full bg-indigo-600 text-white px-6 py-3 rounded-xl font-semibold text-sm hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center justify-center gap-2">
                        {loading
                            ? <><span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" /> Running Analysis...</>
                            : <><Play className="h-4 w-4" /> Run Deep Analysis</>}
                    </button>
                </div>
            ) : (
                <div className="space-y-4">
                    {/* Section tabs */}
                    <div className="flex overflow-x-auto gap-1 bg-slate-100 p-1 rounded-xl">
                        {sections.map(({ id, label, icon: Icon }) => (
                            <button key={id} onClick={() => setActiveSection(id)}
                                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors
                  ${activeSection === id ? 'bg-white text-indigo-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}>
                                <Icon className="h-3.5 w-3.5" />
                                {label}
                            </button>
                        ))}
                    </div>

                    {activeSection === 'stats' && <StatsSection data={data.statistics} />}
                    {activeSection === 'domain' && <DomainSection data={data.domain_analysis} />}
                    {activeSection === 'correlation' && <CorrelationSection data={data.correlations} />}
                    {activeSection === 'univariate' && <UnivariateSection data={data.univariate} />}
                    {activeSection === 'bivariate' && <BivariateSection data={data.bivariate} />}

                    <button onClick={handleAnalyze} disabled={loading}
                        className="w-full bg-slate-100 text-slate-700 px-6 py-3 rounded-xl font-semibold text-sm hover:bg-slate-200 transition-colors">
                        Re-run Analysis
                    </button>
                </div>
            )}
        </div>
    )
}

function StatsSection({ data }) {
    if (!data?.columns) return null
    return (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3 bg-slate-50 border-b text-sm font-medium text-slate-700 flex items-center gap-2">
                <Activity className="h-4 w-4 text-indigo-500" /> Statistical Summary
            </div>
            <div className="overflow-x-auto">
                <table className="text-xs w-full">
                    <thead className="bg-slate-50 border-b border-slate-200">
                        <tr>
                            {['Column', 'Type', 'Missing %', 'Unique', 'Mean', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis'].map(h => (
                                <th key={h} className="px-3 py-2 text-left font-medium text-slate-600 whitespace-nowrap">{h}</th>
                            ))}
                        </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                        {data.columns.map((col) => (
                            <tr key={col.column} className={col.is_target ? 'bg-indigo-50' : 'hover:bg-slate-50'}>
                                <td className="px-3 py-2 font-medium text-slate-800 max-w-[120px] truncate">
                                    {col.is_target && <span className="text-indigo-600 mr-1">🎯</span>}{col.column}
                                </td>
                                <td className="px-3 py-2 text-slate-500">{col.dtype}</td>
                                <td className={`px-3 py-2 ${col.missing_pct > 20 ? 'text-red-600 font-semibold' : 'text-slate-700'}`}>{col.missing_pct}%</td>
                                <td className="px-3 py-2 text-slate-700">{col.unique_count}</td>
                                <td className="px-3 py-2 text-slate-700">{col.mean ?? '—'}</td>
                                <td className="px-3 py-2 text-slate-700">{col.std ?? '—'}</td>
                                <td className="px-3 py-2 text-slate-700">{col.min ?? '—'}</td>
                                <td className="px-3 py-2 text-slate-700">{col.max ?? '—'}</td>
                                <td className={`px-3 py-2 ${Math.abs(col.skewness) > 1 ? 'text-amber-600 font-semibold' : 'text-slate-700'}`}>{col.skewness ?? '—'}</td>
                                <td className="px-3 py-2 text-slate-700">{col.kurtosis ?? '—'}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            <div className="px-5 py-3 bg-blue-50 border-t text-xs text-blue-700">
                <strong>Reading guide:</strong> <span className="text-amber-700 font-semibold">Skewness &gt; |1|</span> = skewed distribution (consider log transform).
                High kurtosis = heavy tails / outlier-prone. 🎯 = target column.
            </div>
        </div>
    )
}

function DomainSection({ data }) {
    if (!data) return null
    return (
        <div className="space-y-4">
            <div className={`rounded-xl p-4 border ${data.total_flagged === 0 ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
                <p className={`font-semibold ${data.total_flagged === 0 ? 'text-green-800' : 'text-amber-800'}`}>
                    {data.total_flagged === 0 ? '✅ No major data quality issues detected.' : `⚠️ ${data.total_flagged} column(s) flagged for quality issues.`}
                </p>
            </div>
            {data.flags.map(flag => (
                <div key={flag.column} className={`bg-white border rounded-xl p-4 ${flag.severity === 'High' ? 'border-red-200' : 'border-amber-200'}`}>
                    <div className="flex items-center gap-2 mb-2">
                        <AlertTriangle className={`h-4 w-4 ${flag.severity === 'High' ? 'text-red-500' : 'text-amber-500'}`} />
                        <span className="font-semibold text-slate-800 text-sm">{flag.column}</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${flag.severity === 'High' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'}`}>
                            {flag.severity}
                        </span>
                    </div>
                    <ul className="space-y-1">
                        {flag.issues.map((issue, i) => (
                            <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-slate-400 shrink-0" />
                                {issue}
                            </li>
                        ))}
                    </ul>
                </div>
            ))}
        </div>
    )
}

function CorrelationSection({ data }) {
    if (!data) return null
    const { target_correlations, multicollinearity_warnings } = data
    return (
        <div className="space-y-4">
            {multicollinearity_warnings?.length > 0 && (
                <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
                    <p className="font-semibold text-amber-800 text-sm mb-2">⚠️ Multicollinearity Detected</p>
                    <p className="text-xs text-amber-700 mb-3">
                        The following pairs of independent features are strongly correlated (r ≥ 0.85).
                        This can cause instability in linear models. Consider dropping one from each pair.
                    </p>
                    <div className="space-y-2">
                        {multicollinearity_warnings.map((w, i) => (
                            <div key={i} className="flex items-center justify-between bg-white rounded-lg px-3 py-2 border border-amber-100 text-xs">
                                <span className="font-medium text-slate-800">{w.feature1} ↔ {w.feature2}</span>
                                <span className={`font-bold ${Math.abs(w.correlation) > 0.95 ? 'text-red-600' : 'text-amber-600'}`}>r = {w.correlation}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {target_correlations?.length > 0 && (
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                    <div className="px-5 py-3 bg-slate-50 border-b text-sm font-medium text-slate-700">Feature → Target Correlation (ranked)</div>
                    <div className="h-72 p-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={target_correlations.slice(0, 12)} layout="vertical" margin={{ left: 90, right: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 11 }} />
                                <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 11 }} />
                                <Tooltip formatter={(v) => [v.toFixed(4), 'Correlation']} contentStyle={{ borderRadius: '8px' }} />
                                <ReferenceLine x={0} stroke="#94a3b8" />
                                <Bar dataKey="correlation" radius={[0, 4, 4, 0]}
                                    fill="#6366f1"
                                    label={false} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="px-5 py-3 bg-blue-50 border-t text-xs text-blue-700">
                        <strong>Why this matters:</strong> Features with high absolute correlation have strong linear predictive power.
                        Positive = feature increases with target; Negative = inverse relationship.
                    </div>
                </div>
            )}
        </div>
    )
}

function UnivariateSection({ data }) {
    if (!data) return null
    const [selected, setSelected] = useState(data.numeric[0]?.column || data.categorical[0]?.column || '')
    const numCol = data.numeric.find(c => c.column === selected)
    const catCol = data.categorical.find(c => c.column === selected)

    return (
        <div className="space-y-4">
            <div className="flex flex-wrap gap-1.5">
                {[...data.numeric, ...data.categorical].map(c => (
                    <button key={c.column} onClick={() => setSelected(c.column)}
                        className={`text-xs px-3 py-1.5 rounded-full border transition-colors
              ${selected === c.column ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-300 hover:border-indigo-400'}`}>
                        {c.column}
                    </button>
                ))}
            </div>
            {numCol && (
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                    <div className="px-5 py-3 bg-slate-50 border-b text-sm font-medium text-slate-700">Distribution: {selected}</div>
                    <div className="h-56 p-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={numCol.bins} margin={{ left: 0, right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="range" tick={{ fontSize: 9 }} interval={2} />
                                <YAxis tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ borderRadius: '8px' }} />
                                <Bar dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="px-5 py-3 bg-blue-50 border-t text-xs text-blue-700">
                        <strong>Why this matters:</strong> The distribution shape reveals if the feature is normal, skewed, or bimodal.
                        Non-normal distributions may benefit from log or Box-Cox transformation.
                    </div>
                </div>
            )}
            {catCol && (
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                    <div className="px-5 py-3 bg-slate-50 border-b text-sm font-medium text-slate-700">Value Counts: {selected}</div>
                    <div className="h-56 p-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={catCol.values} margin={{ left: 0, right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="label" tick={{ fontSize: 11 }} />
                                <YAxis tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ borderRadius: '8px' }} />
                                <Bar dataKey="count" fill="#8b5cf6" radius={[3, 3, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    )
}

function BivariateSection({ data }) {
    if (!data?.features?.length) return <p className="text-slate-500 text-sm">No numeric feature data available for bivariate analysis.</p>
    const [selected, setSelected] = useState(data.features[0]?.feature || '')
    const feat = data.features.find(f => f.feature === selected)

    return (
        <div className="space-y-4">
            <div className="flex flex-wrap gap-1.5">
                {data.features.map(f => (
                    <button key={f.feature} onClick={() => setSelected(f.feature)}
                        className={`text-xs px-3 py-1.5 rounded-full border transition-colors
              ${selected === f.feature ? 'bg-indigo-600 text-white border-indigo-600' : 'bg-white text-slate-600 border-slate-300 hover:border-indigo-400'}`}>
                        {f.feature}
                    </button>
                ))}
            </div>
            {feat && (
                <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
                    <div className="px-5 py-3 bg-slate-50 border-b text-sm font-medium text-slate-700">
                        {feat.feature} vs Target
                    </div>
                    <div className="h-64 p-4">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ left: 10, right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="x" name={feat.feature} tick={{ fontSize: 11 }} label={{ value: feat.feature, position: 'insideBottom', offset: -2, fontSize: 11 }} />
                                <YAxis dataKey="y" name="Target" tick={{ fontSize: 11 }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px' }} />
                                <Scatter data={feat.data} fill="#6366f1" opacity={0.6} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="px-5 py-3 bg-blue-50 border-t text-xs text-blue-700">
                        <strong>Why this matters:</strong> Scatter plots reveal the relationship pattern between a feature and the target.
                        A clear trend indicates the feature is a strong predictor. Clusters or non-linear patterns suggest tree-based models.
                    </div>
                </div>
            )}
        </div>
    )
}
