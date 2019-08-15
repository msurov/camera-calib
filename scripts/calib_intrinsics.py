#!/usr/bin/python

import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import re
from os import listdir
from os.path import isfile, join, split, isdir, splitext
import fnmatch
import matplotlib.pyplot as plt
from glob import glob

'''
    Command line arguments
'''
def parse_mask(path):
    if isdir(path):
        return path, r'*'
    return split(args.srcpath)


def parse_shape(value):
    ans = re.match(r'(\d+)[xX,]\s*(\d+)', value)
    if ans is not None:
        w,h = ans.groups()
        return int(w)-1, int(h)-1
    raise argparse.ArgumentTypeError('pattern shape is in incorrect form')


'''
    Pattern recognition
'''
def list_sources(srcdir, imgmask):
    files = glob(join(srcdir, imgmask))
    files = filter(lambda f: isfile(f), files)
    return files


def print_deviations(status, devs, srcfiles):
    indexes = np.arange(0, len(srcfiles))[status]
    devs2 = [np.mean(d) for d in devs[status]]
    print 'The image points deviation:'
    for d, i in sorted(zip(devs2, indexes), reverse=True):
        _,name = split(srcfiles[i])
        print '  %s: %fpx' % (name, d)


def filter_outliers(status, devs, c=0.9):
    indexes = np.arange(0, len(status))[status]
    devs2 = [np.mean(d) for d in devs[status]]
    pairs = sorted(zip(devs2, indexes), reverse=False)

    n = int(len(pairs) * c)
    pairs = pairs[0:n]
    status2 = np.zeros(len(status), dtype=bool)
    for _,i in pairs:
        status2[i] = True
    return status2


def plot_cross(img, pt, size, color=255):
    x,y = pt
    img[y, x-size//2:x+size//2] = color
    img[y-size//2:y+size//2, x] = color


def draw_corners(status, srcfiles, dstfiles, pattern_shape, imgpoints):
    for s, spath, dpath, pts in zip(status, srcfiles, dstfiles, imgpoints):
        img = cv2.imread(spath)
        if s is False:
            cv2.drawChessboardCorners(img, pattern_shape, None, False)
        else:
            cv2.drawChessboardCorners(img, pattern_shape, pts, True)
        cv2.imwrite(dpath, img)


def path_add_prefix(fullpath, prefix):
    path,name = split(fullpath)
    return join(path, prefix + name)


def expand(status, arr, value=None):
    '''
        substitute the 'value' into the 'arr' according to the 'status'
    '''
    result = []
    i = 0
    for s in status:
        if s:
            result.append(arr[i])
            i += 1
        else:
            result.append(value)

    return np.array(result)


def extract_corners(img, pattern_shape):
    # downscale the image
    scale_factor = int(max(*img.shape) / 1000.)
    if scale_factor > 1:
        n = int(np.log(scale_factor) / np.log(2))
    else:
        n = 0

    low_res_img = img
    for i in range(0,n):
        low_res_img = cv2.pyrDown(low_res_img)

    scale_factor = 2**n
    scan_size = 10 * scale_factor

    # extract corners from low-res image
    retval,corners = cv2.findChessboardCorners(low_res_img, pattern_shape, None)
    if corners is None:
        return None

    if not retval:
        return None

    corners = corners[:,0,:]
    corners = corners * scale_factor

    # refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    res = cv2.cornerSubPix(img, corners, (scan_size,scan_size), (-1,-1), criteria)
    if isinstance(res, list):
        corners = res
    return corners


def collect_imgpoints(srcfiles):
    objp = np.zeros((pattern_shape[0] * pattern_shape[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_shape[0],0:pattern_shape[1]].T.reshape(-1,2)

    objpoints = []
    imgpoints = []
    status = []

    for fpath in srcfiles:
        print 'detecting chessboard in ' + split(fpath)[1] + '.. ',
        img = cv2.imread(fpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = extract_corners(gray, pattern_shape)
        if corners is None:
            print 'couldn\'t find pattern!'
            objpoints.append(None)
            imgpoints.append(None)
            status.append(False)
            continue

        objpoints.append(objp)
        imgpoints.append(corners)
        status.append(True)
        print 'ok'

    return np.array(status), np.array(objpoints), np.array(imgpoints)


def reproject_points(objpoints, rvecs, tvecs, cameraMatrix, distCoeffs):
    imgpoints = []

    for objpts, r, t in zip(objpoints, rvecs, tvecs):
        pts,_ = cv2.projectPoints(objpts, r, t, cameraMatrix, distCoeffs)
        N,w,h = pts.shape
        pts = np.reshape(pts, (N, w*h))
        imgpoints.append(pts)

    return imgpoints


def calibrate(status, objpoints, imgpoints, imgshape):
    objpoints2 = np.array(objpoints[status])
    imgpoints2 = np.array(imgpoints[status])

    if len(imgpoints2) < 2:
        raise Exception('Too few input data')

    tolerance, cameraMatrix, distCoeffs, rvecs, tvecs = \
        cv2.calibrateCamera(objpoints2, imgpoints2, imgshape, None, None, flags=cv2.CALIB_ZERO_TANGENT_DIST)  # 

    imgpoints2_repr = reproject_points(objpoints2, rvecs, tvecs, cameraMatrix, distCoeffs)
    deviations = [np.sqrt(np.sum((p1 - p2)**2, axis=1)) for p1,p2 in zip(imgpoints2, imgpoints2_repr)]
    deviations = expand(status, deviations)

    intrinsics = {
        'tolerance': tolerance,
        'cameraMatrix': cameraMatrix,
        'distCoeffs': distCoeffs,
        'resolution': imgshape
    }

    return intrinsics, deviations


def undistort(intrinsics, srcfiles, dstfiles):
    for (spath, dpath) in zip(srcfiles, dstfiles):
        img = cv2.imread(spath)
        img2 = cv2.undistort(img, intrinsics['cameraMatrix'], intrinsics['distCoeffs'])
        cv2.imwrite(dpath, img2)


def get_mean_deviation(status, devs):
    devs2 = devs[status]
    return [np.sqrt(np.sum(d**2, axis=1)) for d in devs2]


template_txt = '''
camera_matrix = %s
camera_distortion = %s
calibration_tolerance = %f
resolution = %s
'''

template_json = '''
{
    "intrinsics": {
        "K": %s,
        "distortion": %s,
        "calibration_tolerance": %f,
        "resolution": [%d, %d]
    }
}
'''


def array2str(arr):
    elems = ['%.20e' % e for e in np.reshape(arr, np.size(arr))]
    return '[' + ', '.join(elems) + ']'


def save_parameters(saveto, intrinsics):
    p = (
        array2str(intrinsics['cameraMatrix']),
        array2str(intrinsics['distCoeffs']),
        intrinsics['tolerance'],
        intrinsics['resolution'][0], intrinsics['resolution'][1]
    )
    _,ext = splitext(saveto)
    if ext == '.txt':
        data = template_txt % p
    elif ext == '.json':
        data = template_json % p
    else:
        raise Exception('unsupported output format')

    f = open(saveto, 'w+')
    f.write(data)
    f.close()


def print_calib_result(intrinsics):
    colored_green = '\033[92m'
    colored_end = '\033[0m'

    print colored_green
    print '-------------------'
    print 'calibration results'
    print '-------------------'
    print 'tolerance: ', intrinsics['tolerance']
    print 'distortion coefs: '
    print intrinsics['distCoeffs']
    print 'camera matrix: '
    print intrinsics['cameraMatrix']
    print '-------------------'
    print colored_end


def main(srcdir, pattern_shape, imgmask, outdir=None, iterations=1, saveto=None):
    np.set_printoptions(precision=8, suppress=True)

    srcfiles = list_sources(srcdir, imgmask)
    if len(srcfiles) < 2:
        raise Exception('The input folder does not contain photos')

    imshape = cv2.imread(srcfiles[0], 0).shape
    status, objpoints, imgpoints = collect_imgpoints(srcfiles)

    if outdir is not None:
        print 'drawing corners..',
        dstfiles = [join(outdir, 'corners-' + split(f)[1]) for f in srcfiles]
        draw_corners(status, srcfiles, dstfiles, pattern_shape, imgpoints)
        print 'done'

    print 'calibrating..',
    intrinsics, deviations = calibrate(status, objpoints, imgpoints, imshape)
    print 'ok'

    print_deviations(status, deviations, srcfiles)
    print_calib_result(intrinsics)

    for i in xrange(iterations-1):
        status = filter_outliers(status, deviations)
        print 'outliers: '
        for s,f in zip(status, srcfiles):
            if not s:
                _,name = split(f)
                print ' ', name

        if sum(status) < 2:
            print Warning('can\'t proceed %d iterations ' % iterations)
            break

        print 'calibrating..',
        intrinsics, deviations = calibrate(status, objpoints, imgpoints, imshape)
        print 'ok'

        print_deviations(status, deviations, srcfiles)
        print_calib_result(intrinsics)

    if outdir is not None:
        # print 'drawing corners..',
        # dstfiles = [join(outdir, 'corners-' + split(f)[1]) for f in srcfiles]
        # draw_corners(status, srcfiles, dstfiles, pattern_shape, imgpoints)
        # print 'done'

        print 'undistorting..',
        dstfiles = [join(outdir, 'undistort-' + split(f)[1]) for f in srcfiles]
        undistort(intrinsics, srcfiles, dstfiles)
        print 'done'

    if saveto is not None:
        save_parameters(saveto, intrinsics)


'''
 run as:
    python calib.py --srcpath="c:\temp\calib-pattern\samples\IMG_20160526_010427_*.jpg" --pattern=17x10 --iteration=3
'''

def test_():
    im = cv2.imread('/home/msurov/dev/datasets/calib4/Image__2017-08-16__11-43-38.bmp', 0)
    pat_shape = (17,10)
    pat_shape = (pat_shape[0] - 1, pat_shape[1] - 1)
    corners = extract_corners(im, pat_shape)

    plt.imshow(im, cmap='gray', interpolation='nearest')
    plt.plot(corners[:,0], corners[:,1], 'x', markersize=20)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--srcpath', required=True, help='path to directory containing calibration photos')
    parser.add_argument('--outdir', help='path to directory to save undistorted photos')
    parser.add_argument('--pattern', metavar='WxH', required=True, help='shape of the pattern in the form WxH')
    parser.add_argument('--iterations', type=int, default=1, help='number of calibrate iterations, usually between 1..5')
    parser.add_argument('--saveto', help='path to a file to save the camera intrinsic parameters')

    args = parser.parse_args()
    srcdir, imgmask = parse_mask(args.srcpath)
    outdir = args.outdir
    iterations = args.iterations
    saveto = args.saveto
    pattern_shape = parse_shape(args.pattern)

    main(srcdir=srcdir, pattern_shape=pattern_shape,
        imgmask=imgmask, outdir=outdir,
        iterations=iterations, saveto=saveto)
