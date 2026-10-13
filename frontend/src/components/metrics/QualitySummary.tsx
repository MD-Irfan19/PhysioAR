import React from 'react'
import { Award, CheckCircle, BarChart3, HelpCircle } from 'lucide-react'
import type { RepetitionQuality } from '@/types'

interface QualitySummaryProps {
  quality: RepetitionQuality | null
  repNumber?: number
}

function getScoreColor(score: number): { text: string; bg: string; border: string } {
  if (score >= 80) {
    return {
      text: 'text-emerald-400',
      bg: 'bg-emerald-500/10',
      border: 'border-emerald-500/30',
    }
  }
  if (score >= 60) {
    return {
      text: 'text-amber-400',
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
    }
  }
  return {
    text: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/30',
  }
}

export const QualitySummary: React.FC<QualitySummaryProps> = ({ quality, repNumber }) => {
  if (!quality) {
    return (
      <div className="rounded-xl border border-border bg-panel p-5 shadow-sm space-y-3">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <h2 className="text-base font-semibold text-foreground flex items-center gap-2">
            <Award className="h-4 w-4 text-accent" />
            Completed Repetition Quality
          </h2>
          <span className="text-xs text-foreground-subtle">Awaiting Repetition</span>
        </div>
        <div className="flex flex-col items-center justify-center p-6 text-center text-foreground-muted space-y-2">
          <BarChart3 className="h-8 w-8 text-foreground-subtle" />
          <p className="text-xs text-foreground-subtle max-w-xs">
            Complete at least one full repetition through the movement cycle to calculate movement quality scores.
          </p>
        </div>
      </div>
    )
  }

  const overall =
    typeof quality.overall_score === 'number' && !isNaN(quality.overall_score)
      ? quality.overall_score
      : 0
  const colors = getScoreColor(overall)

  const hasRom = typeof quality.rom_score === 'number' && !isNaN(quality.rom_score)
  const hasAlignment = typeof quality.alignment_score === 'number' && !isNaN(quality.alignment_score)
  const hasStability = typeof quality.stability_score === 'number' && !isNaN(quality.stability_score)
  const hasCompensation = typeof quality.compensation_score === 'number' && !isNaN(quality.compensation_score)

  return (
    <div className="rounded-xl border border-border bg-panel p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-border pb-3">
        <h2 className="text-base font-semibold text-foreground flex items-center gap-2">
          <Award className="h-4 w-4 text-accent" />
          Repetition Quality Score
        </h2>
        {repNumber && (
          <span className="rounded bg-panel-hover px-2 py-0.5 text-xs font-semibold text-foreground-muted border border-border">
            Repetition #{repNumber}
          </span>
        )}
      </div>

      {/* Primary Score Ring */}
      <div className={`flex items-center justify-between rounded-xl p-4 border ${colors.bg} ${colors.border}`}>
        <div>
          <span className="text-xs font-semibold uppercase tracking-wider text-foreground-muted">
            Overall Quality Index
          </span>
          <div className="flex items-baseline gap-1 mt-0.5">
            <span className={`text-3xl font-extrabold font-mono ${colors.text}`}>
              {overall.toFixed(1)}
            </span>
            <span className="text-xs font-semibold text-foreground-muted">/ 100</span>
          </div>
        </div>
        <div className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider border ${colors.border} ${colors.text}`}>
          {overall >= 80 ? 'Optimal' : overall >= 60 ? 'Satisfactory' : 'Needs Focus'}
        </div>
      </div>

      {/* Component Sub-scores */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        {hasRom && (
          <div className="rounded-lg border border-border bg-panel-subtle p-2.5">
            <div className="flex justify-between items-center text-foreground-muted">
              <span>ROM Accuracy</span>
              <span className="font-mono font-bold text-foreground">
                {quality.rom_score!.toFixed(1)}
              </span>
            </div>
            <div className="w-full bg-panel rounded-full h-1 mt-1.5 overflow-hidden">
              <div
                className="bg-accent h-1 rounded-full"
                style={{ width: `${Math.min(quality.rom_score!, 100)}%` }}
              />
            </div>
          </div>
        )}

        {hasAlignment && (
          <div className="rounded-lg border border-border bg-panel-subtle p-2.5">
            <div className="flex justify-between items-center text-foreground-muted">
              <span>Postural Alignment</span>
              <span className="font-mono font-bold text-foreground">
                {quality.alignment_score!.toFixed(1)}
              </span>
            </div>
            <div className="w-full bg-panel rounded-full h-1 mt-1.5 overflow-hidden">
              <div
                className="bg-emerald-400 h-1 rounded-full"
                style={{ width: `${Math.min(quality.alignment_score!, 100)}%` }}
              />
            </div>
          </div>
        )}

        {hasStability && (
          <div className="rounded-lg border border-border bg-panel-subtle p-2.5">
            <div className="flex justify-between items-center text-foreground-muted">
              <span>Path Stability</span>
              <span className="font-mono font-bold text-foreground">
                {quality.stability_score!.toFixed(1)}
              </span>
            </div>
            <div className="w-full bg-panel rounded-full h-1 mt-1.5 overflow-hidden">
              <div
                className="bg-amber-400 h-1 rounded-full"
                style={{ width: `${Math.min(quality.stability_score!, 100)}%` }}
              />
            </div>
          </div>
        )}

        {hasCompensation && (
          <div className="rounded-lg border border-border bg-panel-subtle p-2.5">
            <div className="flex justify-between items-center text-foreground-muted">
              <span>Compensation Index</span>
              <span className="font-mono font-bold text-foreground">
                {quality.compensation_score!.toFixed(1)}
              </span>
            </div>
            <div className="w-full bg-panel rounded-full h-1 mt-1.5 overflow-hidden">
              <div
                className="bg-rose-400 h-1 rounded-full"
                style={{ width: `${Math.min(quality.compensation_score!, 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <p className="text-[10px] text-foreground-subtle text-center">
        * Algorithmic heuristic based on repetition kinematics. Not an FDA/CE-marked diagnostic rating.
      </p>
    </div>
  )
}
