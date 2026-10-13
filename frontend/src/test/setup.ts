import '@testing-library/jest-dom/vitest'
import '@testing-library/jest-dom'

// Mock HTMLMediaElement / Image if needed
class MockImage {
  src = ''
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
}
// @ts-expect-error test mock
globalThis.Image = MockImage
