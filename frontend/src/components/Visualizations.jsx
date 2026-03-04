import { useState } from 'react'
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    Cell, LineChart, Line, ScatterChart, Scatter, ReferenceLine, Legend
} from 'recharts'
import { BarChart2, Brain, Lightbulb, TrendingUp, Activity, Target } from 'lucide-react'

const COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#84cc16']

export default function Visualizations({ data, proMode }) {
    if (!data) return null

    const { charts, insight, results, feature_names } = data
    const bestModel = results?.[0]

    const featureImportanceData = bestModel?.feature_importance && feature_names
        ? bestModel.feature_importance
            .map(item => ({
                name: typeof item === 'object' ? (item.feature?.length > 16 ? item.feature.slice(0, 16) + '…' : item.feature) : 'Unknown',
                importance: typeof item === 'object' ? item.importance : item
            }))
            .slice(0, 10)
        : null

    const modelScoreData = results?.map(r => ({
        name: r.model_name.replace(' Regressor', '').replace(' Regression', ''),
        score: r.metrics['Accuracy'] ?? r.metrics['R2 Score'] ?? 0,
        auc: r.metrics['ROC-AUC'] ?? null,
    }))

    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-3">
                    Visualizations & Business Insight
                    {proMode && <span className="text-xs bg-indigo-100 text-indigo-700 px-2.5 py-1 rounded-full font-bold shadow-sm">PRO</span>}
                </h1>
                <p className="mt-1 text-slate-500">Charts, explanations, and a deep domain analysis of your results.</p>
            </div>

            {/* ── Deep Insight Section ─────────────────────────────────── */}
            {insight && typeof insight === 'object' && (
                <InsightSection insight={insight} />
            )}

            {/* ── Target Distribution ──────────────────────────────────── */}
            {charts?.target_distribution?.data && (
                <ChartCard title="Target Distribution" icon={<Target className="h-4 w-4 text-slate-500" />}
                    explanation={charts.target_distribution.explanation}>
                    <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.target_distribution.data} margin={{ right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                                <YAxis tick={{ fontSize: 11 }} />
                                <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0' }} />
                                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                                    {charts.target_distribution.data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── Feature Correlation ───────────────────────────────────── */}
            {charts?.correlation?.data && (
                <ChartCard title="Feature Correlation with Target" icon={<TrendingUp className="h-4 w-4 text-slate-500" />}
                    explanation={charts.correlation.explanation}>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={charts.correlation.data} layout="vertical" margin={{ left: 90, right: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis type="number" domain={[-1, 1]} tick={{ fontSize: 11 }} />
                                <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 10 }} />
                                <Tooltip formatter={(v) => [v.toFixed(4), 'Correlation']} contentStyle={{ borderRadius: '8px' }} />
                                <ReferenceLine x={0} stroke="#94a3b8" />
                                <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                                    {charts.correlation.data.map((item, i) => (
                                        <Cell key={i} fill={item.correlation >= 0 ? '#6366f1' : '#ef4444'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── Feature Importance ────────────────────────────────────── */}
            {featureImportanceData && (
                <ChartCard title={`Feature Importance — ${bestModel.model_name}`} icon={<Brain className="h-4 w-4 text-slate-500" />}
                    explanation="Feature importance shows which variables the model relied on most. High-importance features are strong predictors — they should be closely monitored in production and scrutinized for business relevance.">
                    <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={featureImportanceData} layout="vertical" margin={{ left: 90, right: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis type="number" tick={{ fontSize: 11 }} />
                                <YAxis dataKey="name" type="category" width={90} tick={{ fontSize: 10 }} />
                                <Tooltip contentStyle={{ borderRadius: '8px' }} />
                                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                                    {featureImportanceData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── SHAP Explainability (Pro Mode) ─────────────────────────── */}
            {proMode && bestModel?.shap_plot && (
                <ChartCard title={`Advanced Explainability (SHAP) — ${bestModel.model_name}`} icon={<Activity className="h-4 w-4 text-indigo-500" />}
                    explanation="SHAP (SHapley Additive exPlanations) values break down how each feature contributed to the model's output across all predictions. This goes beyond simple importance by showing the direction (positive/negative impact) and distribution of the effects.">
                    <div className="flex justify-center p-4 bg-slate-50 rounded-xl border border-slate-100">
                        <img
                            src={`data:image/png;base64,${bestModel.shap_plot}`}
                            alt="SHAP Summary Plot"
                            className="max-w-full h-auto rounded-lg shadow-sm"
                        />
                    </div>
                </ChartCard>
            )}

            {/* ── Model Comparison ─────────────────────────────────────── */}
            {modelScoreData && (
                <ChartCard title="Model Performance Comparison" icon={<BarChart2 className="h-4 w-4 text-slate-500" />}
                    explanation="Comparing all trained models helps select the best performer. The gold bar is the top model. Use this chart to decide which model to deploy.">
                    <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={modelScoreData} margin={{ right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                                <Tooltip formatter={(v, n) => [v.toFixed(4), n]} contentStyle={{ borderRadius: '8px' }} />
                                <Legend />
                                <Bar name="Score" dataKey="score" radius={[4, 4, 0, 0]}>
                                    {modelScoreData.map((_, i) => <Cell key={i} fill={i === 0 ? '#f59e0b' : COLORS[i % COLORS.length]} />)}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── ROC-AUC Curve (Classification) ───────────────────────── */}
            {bestModel?.roc_curve && (
                <ChartCard title={`ROC-AUC Curve — ${bestModel.model_name} (AUC = ${bestModel.roc_curve.auc})`}
                    icon={<Activity className="h-4 w-4 text-slate-500" />}
                    explanation="The ROC curve shows the trade-off between True Positive Rate (sensitivity) and False Positive Rate at various thresholds. AUC closer to 1.0 = better discriminative ability. The diagonal dashed line represents random guessing (AUC = 0.5).">
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                                data={bestModel.roc_curve.fpr.map((fpr, i) => ({ fpr, tpr: bestModel.roc_curve.tpr[i] }))}
                                margin={{ left: 10, right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="fpr" type="number" domain={[0, 1]} name="FPR" tick={{ fontSize: 11 }} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                                <Tooltip formatter={(v) => [v.toFixed(4)]} contentStyle={{ borderRadius: '8px' }} />
                                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />
                                <Line type="monotone" dataKey="tpr" stroke="#6366f1" strokeWidth={2.5} dot={false} name="ROC Curve" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── Actual vs Predicted (Regression) ─────────────────────── */}
            {bestModel?.actual_vs_predicted && (
                <ChartCard title={`Actual vs Predicted — ${bestModel.model_name}`}
                    icon={<Activity className="h-4 w-4 text-slate-500" />}
                    explanation="Points close to the 45° diagonal line indicate accurate predictions. A tight cluster along the diagonal means the model generalises well. Scattered points reveal systematic errors or heteroscedasticity.">
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ left: 10, right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="actual" name="Actual" tick={{ fontSize: 11 }} label={{ value: 'Actual', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                                <YAxis dataKey="predicted" name="Predicted" tick={{ fontSize: 11 }} label={{ value: 'Predicted', angle: -90, position: 'insideLeft', fontSize: 11 }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ borderRadius: '8px' }}
                                    formatter={(v, n) => [typeof v === 'number' ? v.toFixed(4) : v, n]} />
                                <Scatter data={bestModel.actual_vs_predicted} fill="#6366f1" opacity={0.6} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── Residual Plot (Regression) ────────────────────────────── */}
            {bestModel?.residuals && (
                <ChartCard title={`Residual Plot — ${bestModel.model_name}`}
                    icon={<Activity className="h-4 w-4 text-slate-500" />}
                    explanation="Residual plot checks if model errors are randomly distributed. Randomly scattered residuals around zero = good fit. Patterns (funnel, curve) indicate violations of model assumptions like non-linearity or heteroscedasticity.">
                    <div className="h-56">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={bestModel.residuals.slice(0, 60)} margin={{ right: 10 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                                <XAxis dataKey="index" tick={{ fontSize: 10 }} label={{ value: 'Sample Index', position: 'insideBottom', offset: -2, fontSize: 11 }} />
                                <YAxis tick={{ fontSize: 11 }} />
                                <ReferenceLine y={0} stroke="#94a3b8" />
                                <Tooltip formatter={(v) => [v.toFixed(4), 'Residual']} contentStyle={{ borderRadius: '8px' }} />
                                <Bar dataKey="residual" radius={[2, 2, 0, 0]}>
                                    {bestModel.residuals.slice(0, 60).map((r, i) => (
                                        <Cell key={i} fill={r.residual >= 0 ? '#6366f1' : '#ef4444'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </ChartCard>
            )}

            {/* ── Confusion Matrix (Pro Mode / Classification) ──────────────── */}
            {proMode && bestModel?.confusion_matrix && (
                <ChartCard title={`Confusion Matrix — ${bestModel.model_name}`} icon={<Target className="h-4 w-4 text-indigo-500" />}
                    explanation="The confusion matrix shows exactly where the model is making mistakes. Rows represent actual classes, columns represent predicted classes. High values on the diagonal = correct predictions. Off-diagonal values reveal which classes are being confused (e.g. False Positives and False Negatives).">
                    <ConfusionMatrix data={bestModel.confusion_matrix} />
                </ChartCard>
            )}

            {/* ── Box Plot Outlier Summary ─────────────────────────────── */}
            {charts?.box_plot?.data && (
                <ChartCard title="Feature Outlier Summary (IQR)" icon={<BarChart2 className="h-4 w-4 text-slate-500" />}
                    explanation={charts.box_plot.explanation}>
                    <div className="overflow-x-auto">
                        <table className="text-xs w-full">
                            <thead className="bg-slate-50 border-b border-slate-200">
                                <tr>
                                    {['Feature', 'Min', 'Q1', 'Median', 'Q3', 'Max', 'IQR'].map(h => (
                                        <th key={h} className="px-3 py-2 text-left font-medium text-slate-600">{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-100">
                                {charts.box_plot.data.map(row => (
                                    <tr key={row.column} className="hover:bg-slate-50">
                                        <td className="px-3 py-2 font-medium text-slate-800">{row.column}</td>
                                        <td className="px-3 py-2 text-slate-700">{row.min}</td>
                                        <td className="px-3 py-2 text-slate-700">{row.q1}</td>
                                        <td className="px-3 py-2 text-slate-700">{row.median}</td>
                                        <td className="px-3 py-2 text-slate-700">{row.q3}</td>
                                        <td className="px-3 py-2 text-slate-700">{row.max}</td>
                                        <td className="px-3 py-2 text-indigo-700 font-semibold">{row.iqr}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </ChartCard>
            )}
        </div>
    )
}

function InsightSection({ insight }) {
    const dimensions = [
        { key: 'what', label: '🔍 What', color: 'indigo', hint: 'What the model predicts' },
        { key: 'how', label: '⚙️ How', color: 'blue', hint: 'How to operationalize it' },
        { key: 'why', label: '🧠 Why', color: 'violet', hint: 'Why this model won' },
        { key: 'where', label: '📍 Where', color: 'cyan', hint: 'Domain & deployment context' },
    ]
    const impacts = insight.impact || []

    return (
        <div className="space-y-4">
            <div className="flex items-center gap-3">
                <div className="h-10 w-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center">
                    <Lightbulb className="h-5 w-5 text-white" />
                </div>
                <div>
                    <h2 className="font-bold text-slate-900">Deep Business Insight</h2>
                    <p className="text-xs text-slate-500">{insight.domain} · {insight.model_name}</p>
                </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {dimensions.map(({ key, label, hint }) => (
                    insight[key] && (
                        <div key={key} className="bg-white border border-slate-200 rounded-xl p-4 space-y-2">
                            <div className="flex items-center justify-between">
                                <span className="font-bold text-slate-800 text-sm">{label}</span>
                                <span className="text-xs text-slate-400">{hint}</span>
                            </div>
                            <p className="text-sm text-slate-600 leading-relaxed"
                                dangerouslySetInnerHTML={{ __html: insight[key].replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                        </div>
                    )
                ))}
            </div>

            {impacts.length > 0 && (
                <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl p-5 space-y-3">
                    <p className="font-bold text-white text-sm">💼 Business Impact</p>
                    <ul className="space-y-2.5">
                        {impacts.map((point, i) => (
                            <li key={i} className="flex items-start gap-2.5 text-sm text-slate-300">
                                <span className="mt-1 h-1.5 w-1.5 rounded-full bg-indigo-400 shrink-0" />
                                <span dangerouslySetInnerHTML={{ __html: point.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>') }} />
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    )
}

function ChartCard({ title, icon, explanation, children }) {
    return (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-5 py-3 border-b border-slate-100 bg-slate-50 flex items-center gap-2">
                {icon}
                <span className="text-sm font-medium text-slate-700">{title}</span>
            </div>
            <div className="p-5">
                {children}
                {explanation && (
                    <div className="mt-4 bg-blue-50 border border-blue-100 rounded-lg p-3">
                        <p className="text-xs text-blue-700"><strong>Why this matters:</strong> {explanation}</p>
                    </div>
                )}
            </div>
        </div>
    )
}

function ConfusionMatrix({ data }) {
    const { matrix, labels } = data
    if (!matrix || !labels) return null

    // Find max value in matrix for normalization of colors
    const maxVal = Math.max(...matrix.flat())

    return (
        <div className="overflow-x-auto p-2">
            <div className="min-w-[400px]">
                {/* Column Labels */}
                <div className="flex ml-24">
                    {labels.map((L, i) => (
                        <div key={i} className="flex-1 text-center py-2 text-[10px] font-bold text-slate-500 uppercase truncate px-1">
                            Pred: {L}
                        </div>
                    ))}
                </div>

                {matrix.map((row, i) => (
                    <div key={i} className="flex h-12">
                        {/* Row Labels */}
                        <div className="w-24 flex items-center justify-end pr-4 text-[10px] font-bold text-slate-500 uppercase text-right leading-tight">
                            Actual: {labels[i]}
                        </div>
                        {row.map((val, j) => {
                            const intensity = maxVal > 0 ? val / maxVal : 0
                            const isDiagonal = i === j
                            return (
                                <div key={j}
                                    className={`flex-1 flex items-center justify-center border border-white relative group transition-colors`}
                                    style={{
                                        backgroundColor: isDiagonal
                                            ? `rgba(99, 102, 241, ${0.1 + intensity * 0.9})` // Indigo for diagonal
                                            : `rgba(239, 68, 68, ${intensity * 0.8})`,     // Red for errors
                                        color: intensity > 0.5 ? '#fff' : '#1e293b'
                                    }}
                                >
                                    <span className="text-xs font-bold">{val}</span>
                                    {/* Tooltip on hover */}
                                    <div className="absolute hidden group-hover:block z-20 bottom-full mb-2 px-2 py-1 bg-slate-800 text-white text-[10px] rounded shadow-lg whitespace-nowrap">
                                        {labels[i]} \u2192 {labels[j]}: {val}
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                ))}
            </div>
            <div className="mt-4 flex justify-center gap-6 text-[10px] font-medium text-slate-500">
                <div className="flex items-center gap-1.5"><span className="h-3 w-3 rounded bg-indigo-500" /> Correct Predictions</div>
                <div className="flex items-center gap-1.5"><span className="h-3 w-3 rounded bg-red-400" /> Errors / Confusion</div>
            </div>
        </div>
    )
}
