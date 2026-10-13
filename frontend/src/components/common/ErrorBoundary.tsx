import React, { Component, ErrorInfo, ReactNode } from 'react'
import { AlertCircle, RotateCcw } from 'lucide-react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error in React component tree:', error, errorInfo)
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null })
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <div className="rounded-xl border border-status-error/40 bg-status-error/10 p-6 text-foreground m-4 space-y-4">
          <div className="flex items-center gap-3 text-status-error">
            <AlertCircle className="h-6 w-6 shrink-0" />
            <h2 className="text-lg font-bold">Dashboard Interface Rendering Error</h2>
          </div>
          <p className="text-xs text-foreground-muted">
            An unexpected error occurred while rendering the exercise interface. Your background
            camera session remains protected.
          </p>
          {this.state.error && (
            <pre className="bg-black/40 border border-status-error/20 p-3 rounded-lg text-xs font-mono text-status-error overflow-auto max-h-40">
              {this.state.error.message}
            </pre>
          )}
          <button
            type="button"
            onClick={this.handleReset}
            className="flex items-center gap-2 rounded-lg bg-panel-hover border border-border px-4 py-2 text-xs font-semibold text-accent hover:border-accent transition-colors"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Reset View & Retry</span>
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
