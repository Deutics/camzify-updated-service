from queue import Queue


class PositionsRecord:
    def __init__(self, size=30, positions=None):
        if positions is None:
            positions = Queue(maxsize=size)
        self._size = size
        self._positions = positions

    def add_position(self, center):
        """*****************************
        Functionality: stores the track in tracks list
        Parameters: centers(x,y)-> center of rectangle
        returns: None
        ********************************"""
        try:
            self._positions.put(center, block=False)
        except:
            self._positions.get()
            self._positions.put(center, block=False)

    def first_position(self):
        """***********************
        Returns the first
        **************************"""
        return self._positions.queue[0]

    def last_position(self):
        """***********************
        Returns the last track
        **************************"""
        return self._positions.queue[-1]

    @property  # getter for positions
    def positions(self):
        return self._positions
