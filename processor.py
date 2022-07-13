import astropy.table as atpy
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import astropy.coordinates as acoo


def mccon_finder(tab, name):
    return np.array([_.rstrip() for _ in tab['GalaxyName']]) == name


mv_sun = 4.2

# mass met rela  from simon 2019
# feh = -1.68 + .28 * (log10(L/Lsun)-6)


# mdf width  from https://iopscience.iop.org/article/10.1088/0004-637X/727/2/78
# sigma = 0.45 -0.06 * log10(l)
def matched_filter_range(xs, cen, sig, minv, maxv, ntarg):
    bin = 0.1
    hh, loc = np.histogram(xs,
                           range=[minv, maxv],
                           bins=int((maxv - minv) / .1))
    xloc = loc[:-1] + .5 * np.diff(loc)
    N = scipy.stats.norm(cen, sig)
    hh1 = N.pdf(xloc)
    rat = hh1 / np.maximum(hh, 1)
    ratmax = rat.max()
    thresholds = np.linspace(0, ratmax, 10)
    logps = np.zeros(len(thresholds))
    edges1 = logps * 0
    edges2 = logps * 0

    for i, curt in enumerate(thresholds):
        xind = rat > curt
        if xind.sum() == 0:
            continue
        xind = np.nonzero(xind)[0]
        x1, x2 = xind[0], xind[-1]
        left = loc[x1]
        right = loc[x2 + 1]
        ncontam = ((xs > left) & (xs < right)).sum()
        nobj = ntarg * (N.cdf(right) - N.cdf(left))
        logp = scipy.stats.poisson(ncontam).logsf(nobj + ncontam)
        logps[i] = logp
        edges1[i] = left
        edges2[i] = right
    pos = np.argmin(logps)
    return (edges1[pos], edges2[pos])


def trier(fehs, mv):
    nstars0 = len(fehs)
    mvs = []
    nums = []
    enums = []
    log10l = (mv_sun - mv) / 2.5
    for propmv in np.linspace(mv, 0, 10):
        proplogl = (-(propmv - mv_sun) / 2.5)
        rat = 10**(proplogl - log10l)
        propfehmean = -1.68 + .28 * (proplogl - 6)
        propfehsig = 0.45 - 0.06 * proplogl
        propexpn0 = (rat * nstars0)
        mvs.append(propmv)
        # print('x', propfehmean)
        left, right = matched_filter_range(fehs, propfehmean, propfehsig, -4,
                                           1, propexpn0)
        nobs = ((fehs < right) & (fehs > left)).sum()
        N = scipy.stats.norm(propfehmean, propfehsig)
        propexpn = propexpn0 * (N.cdf(right) - N.cdf(left))
        # print(propexpn, nobs)
        G = scipy.stats.gamma(nobs + 1, scale=1. / propexpn)
        n1, en1 = G.ppf(.16), G.ppf(.84)
        # print(propexpn, nobs, n1, en1)
        # n1 = nobs*1./(propexpn)
        # en1 = np.sqrt(nobs)/(propexpn)
        nums.append(n1)
        enums.append(en1)

        # 1./0
    return np.array(mvs), np.array(nums), np.array(enums)


tab = atpy.Table().read('vizier_votable.vot')
# https://ui.adsabs.harvard.edu/?#abs/2010ApJS..191..352K
xt = atpy.Table().read('NearbyGalaxies_Jan2021_PUBLIC.fits')
C = acoo.SkyCoord(ra=xt['RA'], dec=xt['Dec'], unit=['hour', 'deg'])
xt['ra'] = C.ra.deg
xt['dec'] = C.dec.deg

maps = [('CanesVenatici(1)', 'CVnI'), ('Draco', 'Dra'), ('Fornax', 'For'),
        ('Leo1', 'LeoI'), ('Leo2', 'LeoII'), ('Sculptor', 'Scl'),
        ('Sextans(1)', 'Sex'), ('UrsaMinor', 'UMi')]

rh = xt['rh'] / 60  # in deg
MV = xt['Vmag'] - xt['dmod']
log10l = 10**(-1. / 2.5 * (MV - 4.2))
fehs = tab['__Fe_H_']
cnt = 0
plt.clf()
for n_mc, n_kirb in maps:
    #xind1 = xt['GalaxyName']==n_mc
    xind1 = mccon_finder(xt, n_mc)
    curmv = float(MV[xind1])
    curlogl = float(log10l[xind1])

    print(curmv)
    xind2 = tab['dSph'] == n_kirb
    assert (xind2.sum() > 0)

    xmv, xn, xen = trier(fehs[xind2], curmv)
    plt.subplot(3, 3, cnt + 1)
    plt.title(n_kirb)
    plt.fill_between(xmv, xn, xen)
    plt.gca().set_yscale('log')
    if cnt > 6:
        plt.xlabel('$M_V$')
    if cnt % 3 == 0:
        plt.ylabel('N(merged max)')
    cnt += 1
plt.gcf().set_size_inches(7, 7)
plt.tight_layout()
plt.savefig('dwarf_dwarf.pdf')
