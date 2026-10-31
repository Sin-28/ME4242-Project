import sys
import curses
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from emio_arrow_control import (parse_args, goal_pulses, keyboard_loop,
                                load_recording, numbered_motion_path, save_recording)


class ArrowConfigurationTests(unittest.TestCase):
    def test_new_defaults(self):
        with patch.object(sys, 'argv', ['emio_arrow_control.py']):
            args, mapping = parse_args()
        self.assertEqual(args.step, 5.0)
        self.assertEqual(mapping, {'up': 0, 'down': 2, 'left': 1, 'right': 3})
        self.assertEqual(args.reverse, [1, 3])

    def test_horizontal_motor_direction_is_reversed(self):
        mapping = {'up': 0, 'down': 2, 'left': 1, 'right': 3}
        baseline = [2048] * 4
        self.assertGreater(goal_pulses(baseline, -5, 0, mapping, {1, 3}, 100)[1], 2048)
        self.assertGreater(goal_pulses(baseline, 5, 0, mapping, {1, 3}, 100)[3], 2048)

    def test_goal_can_cross_single_turn_boundary(self):
        mapping = {'up': 0, 'down': 2, 'left': 1, 'right': 3}
        goals = goal_pulses([2048] * 4, 0, 270, mapping, set(), 4096/(2*3.1416))
        self.assertEqual(goals[0], -1024)
        self.assertEqual(goals[1:], [2048] * 3)

    def test_extended_goal_can_be_negative(self):
        mapping = {'up': 0, 'down': 2, 'left': 1, 'right': 3}
        goals = goal_pulses([100] * 4, 0, 180, mapping, set(), 4096/(2*3.1416))
        self.assertEqual(goals, [-1948, 100, 100, 100])

    def test_recording_round_trip_without_angle_limit(self):
        with patch.object(sys, 'argv', ['emio_arrow_control.py']):
            args, mapping = parse_args()
        events = [{'time': 0.25, 'x': 720.0, 'y': 0.0},
                  {'time': 1.0, 'x': 0.0, 'y': 0.0}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion.json'
            save_recording(path, mapping, [1, 3], events)
            self.assertEqual(load_recording(path, args, mapping), (events, mapping, [1, 3]))

    def test_old_recording_keeps_its_original_motor_configuration(self):
        with patch.object(sys, 'argv', ['emio_arrow_control.py']):
            args, mapping = parse_args()
        old_mapping = {'up': 0, 'down': 2, 'left': 1, 'right': 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old.json'
            save_recording(path, old_mapping, [], [{'time': 0.2, 'x': 5, 'y': 0}])
            self.assertEqual(load_recording(path, args, mapping),
                             ([{'time': 0.2, 'x': 5, 'y': 0}], old_mapping, []))

    def test_session_records_relative_to_current_pose_and_keeps_connection_open(self):
        with patch.object(sys, 'argv', ['emio_arrow_control.py', '--session']):
            args, mapping = parse_args()
        class Screen:
            keys = iter([curses.KEY_RIGHT, ord('r'), ord('2'), curses.KEY_LEFT,
                         ord('r'), ord('q')])
            def keypad(self, enabled): pass
            def timeout(self, value): pass
            def getch(self): return next(self.keys)
        positions = [2048] * 4
        motors = SimpleNamespace(_rad_to_pulse=100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion_2.json'
            with patch('emio_arrow_control.curses.curs_set'), \
                 patch('emio_arrow_control.draw'), \
                 patch('emio_arrow_control.check_motor_health'), \
                 patch('emio_arrow_control.numbered_motion_path', return_value=path), \
                 patch('emio_arrow_control.read_positions', side_effect=lambda _: list(positions)), \
                 patch('emio_arrow_control.write_goals', side_effect=lambda _, goals: positions.__setitem__(slice(None), goals)):
                keyboard_loop(Screen(), motors, [2048] * 4, args, mapping)
            events, _, _ = load_recording(path, args, mapping)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['x'], -5)

    def test_number_key_selects_existing_motion_filename(self):
        self.assertEqual(numbered_motion_path(ord('1')), Path('motion_1.json'))
        self.assertEqual(numbered_motion_path(ord('2')), Path('motion_2.json'))
        self.assertIsNone(numbered_motion_path(ord('r')))

    def test_session_plays_saved_motor_configuration_then_returns_to_manual(self):
        with patch.object(sys, 'argv', ['emio_arrow_control.py', '--session']):
            args, mapping = parse_args()
        class Screen:
            keys = iter([ord('p'), ord('1'), -1, -1, ord('q')])
            def keypad(self, enabled): pass
            def timeout(self, value): pass
            def getch(self): return next(self.keys)
        positions = [2048] * 4
        sent = []
        def write(_, goals):
            positions[:] = goals
            sent.append(list(goals))
        motors = SimpleNamespace(_rad_to_pulse=100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'motion_1.json'
            save_recording(path, mapping, [], [{'time': 0, 'x': 5, 'y': 0}])
            with patch('emio_arrow_control.curses.curs_set'), \
                 patch('emio_arrow_control.draw'), \
                 patch('emio_arrow_control.check_motor_health'), \
                 patch('emio_arrow_control.numbered_motion_path', return_value=path), \
                 patch('emio_arrow_control.read_positions', side_effect=lambda _: list(positions)), \
                 patch('emio_arrow_control.write_goals', side_effect=write):
                keyboard_loop(Screen(), motors, [2048] * 4, args, mapping)
        self.assertTrue(any(goals[3] < 2048 for goals in sent))


if __name__ == '__main__':
    unittest.main()
