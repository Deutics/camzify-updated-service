import math
from shapely.geometry import LineString


class Line:
    def __init__(self, start_position=None, end_position=None):
        self._start_position = start_position
        self._end_position = end_position

    def find_angle_of_intersection(self, track):
        """**************************************
        Functionality: First finds if line and object intersects and then find the angle of intersecton
        Parameter: track(object of track)
        Returns: angle of intersection in degree
        ****************************************"""
        if self.check_if_intersect(track):
            x1, y1 = track.past_positions.first_position()
            x2, y2 = track.past_positions.last_position()
            x3, y3 = self._start_position
            x4, y4 = self._end_position

            # Angles of lines
            angle_of_line1 = math.atan2(y2 - y1, x2 - x1)
            angle_of_line2 = math.atan2(y4 - y3, x4 - x3)

            # Angle of intersection
            angle_of_intersection = angle_of_line2 - angle_of_line1

            return math.degrees(angle_of_intersection)

    def check_if_intersect(self, track):
        """**************************************
        Functionality: First finds if line and object intersects
        Parameter: track(object of track)
        Returns: True if line and object intersects
        ****************************************"""
        is_intersecting = False

        # checks if the obj has already crossed
        if not track.notification_generated:
            object_position = LineString([track.past_positions.last_position(), track.past_positions.first_position()])

            # checks if intersects
            line_position = LineString([self._start_position, self._end_position])
            if line_position.intersects(object_position):
                is_intersecting = True

        return is_intersecting

    @property
    def start_position(self):
        return self._start_position

    @property
    def end_position(self):
        return self._end_position
