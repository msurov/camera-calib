import matplotlib.pyplot as plt
import argparse


__colors = {
    'black': (0,0,0),
    'white': (255,255,255),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'red': (255, 0, 0),
}


def get_color(c):
    if c in __colors:
        r,g,b = __colors[c]
        return r * 256 * 256 + g * 256 + b
    elif type(c) == tuple:
        r,g,b = c
        return r * 256 * 256 + g * 256 + b
    elif type(c) == int:
        return c
    raise Exception('Can\'t convert color')


def svg_rect(x,y,w,h,c):
    return '''<rect
       style="fill:#%06x;fill-opacity:1;stroke:none;stroke-width:2;stroke-miterlimit:4;stroke-dasharray:none;stroke-dashoffset:0;stroke-opacity:1"
       width="%fmm"
       height="%fmm"
       x="%fmm"
       y="%fmm"/>
    ''' % (get_color(c), w, h, x, y)


def chessboard(nx, ny, squareszmm, p0=[0,0]):
    rects = ''
    x0,y0 = p0

    for y in range(0, ny):
        for x in range(y % 2, nx, 2):
            rects += svg_rect(
                x0 + x * squareszmm,
                y0 + y * squareszmm,
                squareszmm, squareszmm, 'black')

    return '''<g>\n%s\n</g>''' % rects


def svg_doc(content, sizemm):
    w,h = sizemm
    template = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?> 
        <svg
            width="%dmm"
            height="%dmm"
        >
            <rect width="100%%" height="100%%"  fill="white"/>
            %s
        </svg>
    '''
    return template % (w, h, ''.join(content))


def gen_pattern(cfg, filename):
    assert filename.endswith('.svg')

    border = cfg['border']
    nx = cfg['nx']
    ny = cfg['ny']
    sz = cfg['square_size_mm']

    docsz = nx * sz + 2 * border, ny * sz + 2 * border
    p0 = border, border
    cb = chessboard(nx, ny, sz, p0)
    doc = svg_doc([cb], docsz)

    f = open(filename, 'w')
    f.write(doc)
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create chessboard image')
    parser.add_argument('shape', type=str, required=True,
                        help='chape of the pattern as WxH')
    parser.add_argument('border', type=float,
                        help='border size in mm')
    parser.add_argument('square', type=float,
                        help='side of square in mm')
    args = parser.parse_args()

    # cfg = {
    #     'border': 10,
    #     'nx': 16,
    #     'ny': 9,
    #     'square_size_mm': 30
    # }
    # gen_pattern(cfg, 'chessboard.svg')


2000
pontiac aztec