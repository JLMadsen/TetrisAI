"""
Not used currently

For editing current shape

keep track of origo (reference point) of shape
and relative blocks

"""

class Piece:
    
    def __init__(self, shape):

        self.pos = [1, 1]
        self.shape = shape

        self.origo = None
        self.relative = []

        print(shape)

        for i, row in enumerate(shape):
            for j, cell in enumerate(row):

                if cell != '0' and self.origo is None:
                    self.origo = [i, j]

                elif cell != '0':
                    self.relative.append([i - self.origo[0], j - self.origo[1]])

    def all_blocks(self):
        return self.relative + [self.origo]

if __name__ == "__main__":
    from play import main
    main()