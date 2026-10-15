import { render, screen } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { MetricCards } from '@/components/metrics/MetricCards'
import { FeedbackPanel } from '@/components/feedback/FeedbackPanel'
import { QualitySummary } from '@/components/metrics/QualitySummary'
import { VideoFeed } from '@/components/video/VideoFeed'
import { SessionSetup } from '@/components/session/SessionSetup'
import type { SessionSnapshot } from '@/types'

describe('PhysioAR UI Components', () => {
  describe('MetricCards Component', () => {
    it('1. renders dashes ("—") for null/unavailable values rather than misleading zeros', () => {
      const emptySnapshot: SessionSnapshot = {
        status: 'idle',
        exercise_name: null,
        exercise_id: null,
        side: null,
        selected_side: null,
        camera_orientation: null,
        is_calibrated: false,
        calibration_status: 'not_calibrated',
        calibration_error: null,
        calibration_samples: 0,
        calibration_baseline: null,
        pose_detected: false,
        current_angle: null,
        joint_angle: null,
        rep_state: null,
        rep_phase: null,
        rep_count: 0,
        rom_target: null,
        rom_progress: null,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 0,
      }

      render(<MetricCards snapshot={emptySnapshot} />)

      // Angle should be "—"
      const dashes = screen.getAllByText('—')
      expect(dashes.length).toBeGreaterThanOrEqual(2) // Joint angle and ROM progress
      expect(screen.getByText('0')).toBeInTheDocument() // Rep count is 0
    })

    it('2. correctly formats populated joint angles and percentages', () => {
      const activeSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 82.4,
        joint_angle: 82.4,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: 4,
        rom_target: 90.0,
        rom_progress: 0.915,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 120,
      }

      render(<MetricCards snapshot={activeSnapshot} />)

      expect(screen.getByText('82.4')).toBeInTheDocument()
      expect(screen.getByText('4')).toBeInTheDocument()
      expect(screen.getByText('92%')).toBeInTheDocument()
      expect(screen.getByText('Baseline Active')).toBeInTheDocument()
    })

    it('2b. safely handles NaN, non-finite numbers, and negative counts without crashing', () => {
      const corruptSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: false,
        calibration_status: 'calibrating',
        calibration_error: null,
        calibration_samples: NaN,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: NaN,
        joint_angle: Infinity as any,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: -5,
        rom_target: NaN,
        rom_progress: NaN,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 10,
      }

      render(<MetricCards snapshot={corruptSnapshot} />)

      // NaN angles and progress should render graceful dashes
      const dashes = screen.getAllByText('—')
      expect(dashes.length).toBeGreaterThanOrEqual(2)
      // Negative rep count is normalized safely to 0
      expect(screen.getByText('0')).toBeInTheDocument()
      // Calibrating status remains clearly indicated
      expect(screen.getByText('Collecting...')).toBeInTheDocument()
    })

    it('2c. displays large repetition counts and formatted states cleanly', () => {
      const highRepSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Sit to Stand',
        exercise_id: 'sit_to_stand',
        side: 'left',
        selected_side: 'left',
        camera_orientation: 'side',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 110.5,
        joint_angle: 110.5,
        rep_state: 'extended_hold_motion',
        rep_phase: 'extended_hold_motion',
        rep_count: 1250,
        rom_target: 95.0,
        rom_progress: 1.15,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 5000,
      }

      render(<MetricCards snapshot={highRepSnapshot} />)

      expect(screen.getByText('1250')).toBeInTheDocument()
      expect(screen.getByText('110.5')).toBeInTheDocument()
      expect(screen.getByText('Target: 95°')).toBeInTheDocument()
      // ROM progress clamped to 100%
      expect(screen.getByText('100%')).toBeInTheDocument()
      expect(screen.getByText(/State: extended hold motion/i)).toBeInTheDocument()
    })

    it('2d. renders telemetry strip with accessible roles and failure details', () => {
      const failedSnapshot: SessionSnapshot = {
        status: 'error',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: false,
        calibration_status: 'failed',
        calibration_error: 'Patient not centered in frame',
        calibration_samples: 5,
        calibration_baseline: null,
        pose_detected: false,
        current_angle: null,
        joint_angle: null,
        rep_state: null,
        rep_phase: null,
        rep_count: 0,
        rom_target: 90.0,
        rom_progress: 0.45,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 45,
      }

      render(<MetricCards snapshot={failedSnapshot} />)

      // Telemetry strip role & testid
      expect(screen.getByTestId('telemetry-strip')).toBeInTheDocument()
      expect(screen.getByRole('progressbar', { name: /ROM Completion Percentage/i })).toBeInTheDocument()
      expect(screen.getByText('Calibration Failed')).toBeInTheDocument()
      expect(screen.getByText('Patient not centered in frame')).toBeInTheDocument()
    })
  })

  describe('FeedbackPanel Component', () => {
    it('3. renders active compensations with friendly clinical labels and warning cues', () => {
      const compSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 60.0,
        joint_angle: 60.0,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: 1,
        rom_target: 90.0,
        rom_progress: 0.66,
        active_compensation_ids: ['torso_lean', 'shoulder_hike'],
        feedback_messages: ['Keep your shoulders level'],
        latest_quality: null,
        last_error: null,
        frame_index: 80,
      }

      render(<FeedbackPanel snapshot={compSnapshot} />)

      expect(screen.getByText('Torso Lateral Lean')).toBeInTheDocument()
      expect(screen.getByText('Shoulder Hike / Elevation')).toBeInTheDocument()
      expect(screen.getByText('Keep your shoulders level')).toBeInTheDocument()
      expect(screen.getByText(/2 Flagged/i)).toBeInTheDocument()
    })

    it('4. shows positive alignment state when no compensations flagged during active pose', () => {
      const cleanSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 45.0,
        joint_angle: 45.0,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: 0,
        rom_target: 90.0,
        rom_progress: 0.5,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 20,
      }

      render(<FeedbackPanel snapshot={cleanSnapshot} />)

      expect(screen.getByText(/Optimal form maintained/i)).toBeInTheDocument()
    })

    it('4b. preserves all known compensation ID mappings and qualified proxy labels', () => {
      const allCompSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Sit to Stand',
        exercise_id: 'sit_to_stand',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'side',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 90.0,
        joint_angle: 90.0,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: 1,
        rom_target: 95.0,
        rom_progress: 0.9,
        active_compensation_ids: [
          'torso_lean',
          'shoulder_hike',
          'neck_tilt',
          'hip_rotation',
          'lateral_trunk_lean',
          'hip_hike',
          'shoulder_substitution',
          'insufficient_forward_trunk_lean',
          'hand_assisted_rising',
          'asymmetric_weight_shift',
        ],
        feedback_messages: ['Lean trunk forward'],
        latest_quality: null,
        last_error: null,
        frame_index: 100,
      }

      render(<FeedbackPanel snapshot={allCompSnapshot} />)

      // Verified exact mapping labels
      expect(screen.getByText('Torso Lateral Lean')).toBeInTheDocument()
      expect(screen.getByText('Shoulder Hike / Elevation')).toBeInTheDocument()
      expect(screen.getByText('Cervical Spine Tilt')).toBeInTheDocument()
      expect(screen.getByText('Pelvic / Hip Rotation')).toBeInTheDocument()
      expect(screen.getByText('Trunk Lateral Deviation')).toBeInTheDocument()
      expect(screen.getByText('Pelvic Hike')).toBeInTheDocument()
      expect(screen.getByText('Shoulder Substitution')).toBeInTheDocument()
      expect(screen.getByText('Insufficient Forward Trunk Inclination')).toBeInTheDocument()
      // Crucial proxy qualifications
      expect(screen.getByText('Hand-Assisted Rising Proxy')).toBeInTheDocument()
      expect(screen.getByText('Asymmetric Weight Shift Proxy')).toBeInTheDocument()
      expect(screen.getByText(/10 Flagged/i)).toBeInTheDocument()
    })

    it('4c. renders unknown compensation identifiers safely with readable fallback', () => {
      const unknownSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Custom Exercise',
        exercise_id: 'custom_ex',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 50.0,
        joint_angle: 50.0,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: 0,
        rom_target: 90.0,
        rom_progress: 0.5,
        active_compensation_ids: ['excessive_lumbar_lordosis'],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 25,
      }

      render(<FeedbackPanel snapshot={unknownSnapshot} />)

      expect(screen.getByText('Excessive Lumbar Lordosis')).toBeInTheDocument()
    })

    it('4d. prioritizes active compensations before disclaimer and retains clinical disclaimer', () => {
      const activeSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: 60.0,
        joint_angle: 60.0,
        rep_state: 'rising',
        rep_phase: 'rising',
        rep_count: 1,
        rom_target: 90.0,
        rom_progress: 0.66,
        active_compensation_ids: ['shoulder_hike'],
        feedback_messages: ['Lower your shoulder during abduction'],
        latest_quality: null,
        last_error: null,
        frame_index: 80,
      }

      const { container } = render(<FeedbackPanel snapshot={activeSnapshot} />)

      // Both compensation and disclaimer are present
      expect(screen.getByText('Shoulder Hike / Elevation')).toBeInTheDocument()
      expect(screen.getByText('Clinical Disclaimer')).toBeInTheDocument()
      expect(screen.getByText(/PhysioAR is an engineering guidance tool/i)).toBeInTheDocument()

      // Verify DOM ordering: Flagged compensation appears before disclaimer
      const panelText = container.textContent || ''
      const compPos = panelText.indexOf('Shoulder Hike / Elevation')
      const cuePos = panelText.indexOf('Lower your shoulder during abduction')
      const disclaimerPos = panelText.indexOf('Clinical Disclaimer')

      expect(compPos).toBeGreaterThan(-1)
      expect(cuePos).toBeGreaterThan(compPos)
      expect(disclaimerPos).toBeGreaterThan(cuePos)
    })

    it('4e. does NOT show misleading success state when pose is not detected', () => {
      const noPoseSnapshot: SessionSnapshot = {
        status: 'active',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: true,
        calibration_status: 'calibrated',
        calibration_error: null,
        calibration_samples: 30,
        calibration_baseline: null,
        pose_detected: false, // Pose lost!
        current_angle: null,
        joint_angle: null,
        rep_state: null,
        rep_phase: null,
        rep_count: 0,
        rom_target: 90.0,
        rom_progress: null,
        active_compensation_ids: [], // Empty, but pose is missing!
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 40,
      }

      render(<FeedbackPanel snapshot={noPoseSnapshot} />)

      // Must show Subject Not Detected
      expect(screen.getByText('Subject Not Detected')).toBeInTheDocument()
      expect(screen.getByText('No Pose')).toBeInTheDocument()
      // MUST NOT show optimal form maintained!
      expect(screen.queryByText(/Optimal form maintained/i)).not.toBeInTheDocument()
    })

    it('4f. cleanly renders idle and calibrating session states', () => {
      // 1. Idle state
      const { unmount } = render(<FeedbackPanel snapshot={null} />)
      expect(screen.getByText('Session Standby')).toBeInTheDocument()
      expect(screen.getByText('Standby')).toBeInTheDocument()
      unmount()

      // 2. Calibrating state
      const calibSnapshot: SessionSnapshot = {
        status: 'calibrating',
        exercise_name: 'Shoulder Abduction',
        exercise_id: 'shoulder_abduction',
        side: 'right',
        selected_side: 'right',
        camera_orientation: 'front',
        is_calibrated: false,
        calibration_status: 'calibrating',
        calibration_error: null,
        calibration_samples: 10,
        calibration_baseline: null,
        pose_detected: true,
        current_angle: null,
        joint_angle: null,
        rep_state: null,
        rep_phase: null,
        rep_count: 0,
        rom_target: 90.0,
        rom_progress: null,
        active_compensation_ids: [],
        feedback_messages: [],
        latest_quality: null,
        last_error: null,
        frame_index: 10,
      }

      render(<FeedbackPanel snapshot={calibSnapshot} />)
      expect(screen.getByText('Neutral Baseline Calibration')).toBeInTheDocument()
      expect(screen.getByText('Calibrating')).toBeInTheDocument()
    })

    it('4g. handles null, malformed, or missing optional fields without crashing', () => {
      const malformedSnapshot: any = {
        status: 'active',
        pose_detected: true,
        active_compensation_ids: null, // intentionally corrupt nulls
        feedback_messages: null,
        calibration_error: null,
      }

      expect(() => {
        render(<FeedbackPanel snapshot={malformedSnapshot} />)
      }).not.toThrow()
    })
  })

  describe('QualitySummary Component', () => {
    it('5. shows awaiting state when no completed rep exists', () => {
      render(<QualitySummary quality={null} />)
      expect(screen.getByText(/Awaiting Repetition/i)).toBeInTheDocument()
      expect(screen.getByText(/Complete at least one full repetition/i)).toBeInTheDocument()
    })

    it('6. displays component sub-scores when quality data provided', () => {
      const quality = {
        overall_score: 87.5,
        rom_score: 90.0,
        alignment_score: 85.0,
        stability_score: 88.0,
        compensation_score: 85.0,
      }

      render(<QualitySummary quality={quality} repNumber={2} />)

      expect(screen.getByText('87.5')).toBeInTheDocument()
      expect(screen.getByText('Repetition #2')).toBeInTheDocument()
      expect(screen.getByText('ROM Accuracy')).toBeInTheDocument()
      expect(screen.getByText('Postural Alignment')).toBeInTheDocument()
      expect(screen.getByText('Path Stability')).toBeInTheDocument()
      expect(screen.getByText('Compensation Index')).toBeInTheDocument()
    })
  })

  describe('VideoFeed Component', () => {
    it('7. displays standby placeholder when session is idle', () => {
      render(<VideoFeed sessionStatus="idle" snapshot={null} />)
      expect(screen.getByText(/STREAM STANDBY/i)).toBeInTheDocument()
      expect(screen.getByText(/Webcam Stream Idle/i)).toBeInTheDocument()
    })

    it('8. renders image element pointing to stream URL when active', () => {
      render(
        <VideoFeed
          sessionStatus="active"
          snapshot={{
            status: 'active',
            exercise_name: 'Sit to Stand',
            exercise_id: 'sit_to_stand',
            side: 'right',
            selected_side: 'right',
            camera_orientation: 'side',
            is_calibrated: true,
            calibration_status: 'calibrated',
            calibration_error: null,
            calibration_samples: 30,
            calibration_baseline: null,
            pose_detected: true,
            current_angle: 90.0,
            joint_angle: 90.0,
            rep_state: 'up',
            rep_phase: 'up',
            rep_count: 1,
            rom_target: 95.0,
            rom_progress: 0.95,
            active_compensation_ids: [],
            feedback_messages: [],
            latest_quality: null,
            last_error: null,
            frame_index: 60,
          }}
        />
      )

      expect(screen.getByText(/LIVE FEED/i)).toBeInTheDocument()
      expect(screen.getByText(/Pose Detected/i)).toBeInTheDocument()
      const img = screen.getByRole('img')
      expect(img).toHaveAttribute('src', expect.stringContaining('/api/video/stream'))
    })
  })

  describe('SessionSetup Component', () => {
    it('9. disables inputs and shows progress indicator while session is active', () => {
      render(
        <SessionSetup
          exercises={[
            {
              id: 'shoulder_abduction',
              name: 'Shoulder Abduction',
              target_joint: 'shoulder',
              camera_orientation: 'front',
              rom_target: 90.0,
            },
          ]}
          selectedExerciseId="shoulder_abduction"
          selectedSide="right"
          cameraIndex={0}
          sessionStatus="active"
          isLoading={false}
          error={null}
          onSelectExercise={vi.fn()}
          onSelectSide={vi.fn()}
          onChangeCameraIndex={vi.fn()}
          onStart={vi.fn()}
          onStop={vi.fn()}
          onCalibrate={vi.fn()}
        />
      )

      expect(screen.getByRole('button', { name: /Stop Exercise Session/i })).toBeInTheDocument()
      expect(screen.getByRole('button', { name: /Calibrate/i })).toBeInTheDocument()
      expect(screen.getByLabelText(/Exercise Protocol/i)).toBeDisabled()
    })
  })
})
