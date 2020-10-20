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

        print(self.origo)
        print(self.relative)

if __name__ == "__main__":
    from play import main
    main()