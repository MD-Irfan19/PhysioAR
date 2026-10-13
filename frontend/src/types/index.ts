/**
 * PhysioAR TypeScript Type Definitions.
 * Grounded directly in the backend schemas (src/api/schemas.py).
 */

export type SessionStatusType = 'idle' | 'calibrating' | 'active' | 'stopped' | 'error';

export type CalibrationStatusType = 'not_calibrated' | 'calibrating' | 'calibrated' | 'failed';

export interface ExerciseItem {
  id: string;
  name: string;
  target_joint: string;
  camera_orientation: string;
  rom_target: number;
}

export interface ExerciseListResponse {
  exercises: ExerciseItem[];
}

export interface RepetitionQuality {
  overall_score: number;
  rom_score?: number;
  alignment_score?: number;
  stability_score?: number;
  compensation_score?: number;
  [key: string]: any;
}

export interface SessionSnapshot {
  status: SessionStatusType;
  exercise_name: string | null;
  exercise_id: string | null;
  side: string | null;
  selected_side: string | null;
  camera_orientation: string | null;
  is_calibrated: boolean;
  calibration_status: string;
  calibration_error: string | null;
  calibration_samples: number;
  calibration_baseline: Record<string, any> | null;
  pose_detected: boolean;
  current_angle: number | null;
  joint_angle: number | null;
  rep_state: string | null;
  rep_phase: string | null;
  rep_count: number;
  rom_target: number | null;
  rom_progress: number | null;
  active_compensation_ids: string[];
  feedback_messages: string[];
  latest_quality: RepetitionQuality | null;
  last_error: string | null;
  frame_index: number;
}

export interface HealthResponse {
  status: string;
  runtime_running: boolean;
  runtime_error: string | null;
  camera_active: boolean;
  version: string;
}

export interface SessionStatusResponse {
  status: string;
  is_running: boolean;
  exercise_name: string | null;
  exercise_id: string | null;
  side: string | null;
  error: string | null;
}

export interface StartSessionRequest {
  exercise_id: string;
  side: 'left' | 'right';
  camera_index?: number;
}

export interface CalibrateRequest {
  duration_seconds?: number;
  min_samples?: number;
}

export interface GenericActionResponse {
  status: string;
  message: string;
}

export interface WSSnapshotMessage {
  type: 'snapshot';
  timestamp: number;
  data: SessionSnapshot;
}

export interface WSErrorMessage {
  type: 'error';
  timestamp: number;
  error: string;
}

export type WSMessage = WSSnapshotMessage | WSErrorMessage;

export type ConnectionState = 'connected' | 'reconnecting' | 'disconnected';
