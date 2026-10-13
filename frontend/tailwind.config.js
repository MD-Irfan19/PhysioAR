/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0B0F14",
        panel: {
          DEFAULT: "#111827",
          hover: "#1F2937",
          subtle: "#0E1522",
        },
        border: "#1F2937",
        accent: {
          DEFAULT: "#3CCFFF",
          hover: "#22B8E6",
          dim: "rgba(60, 207, 255, 0.15)",
        },
        status: {
          success: "#10B981",
          warning: "#F59E0B",
          error: "#EF4444",
          info: "#3CCFFF",
        },
        foreground: {
          DEFAULT: "#F3F4F6",
          muted: "#9CA3AF",
          subtle: "#6B7280",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          '"Helvetica Neue"',
          "Arial",
          "sans-serif",
        ],
        mono: [
          '"JetBrains Mono"',
          '"Fira Code"',
          "Consolas",
          "monospace",
        ],
      },
      animation: {
        pulseSlow: "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
      },
    },
  },
  plugins: [],
}
