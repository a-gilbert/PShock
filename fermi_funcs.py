"""Accelerated function for evaluating fermi integrals and their inverses
based on fits from [1].

[1] Antia, H. M. (1993). Rational Function Approximations 
for Fermi-Dirac Integrals. The Astrophysical Journal Supplement Series, 
84, 101. https://doi.org/10.1086/191748"""
import numpy as np
from numba import jit


#Fermi Integral Coeffs. Accessed by C[n][order][letter] with
#order in (1,2,3)
#n in (-0.5, 0.5, 1.5, 2.5)
#letter in ('a', 'b', 'c', 'd')'
coeffs = {-0.5:{
                1:{'a':np.array([2.31456e1, 1.37820e1,
                                1.00000], dtype=np.longdouble),
                    'b':np.array([1.30586e1, 1.70048e1,
                                5.07527e0, 2.36620e-1], dtype=np.longdouble),
                    'c':np.array([1.53602e-2, 1.46815e-1, 
                                1.00000e0], dtype=np.longdouble),
                    'd':np.array([7.68015e-3, 7.63700e-2,
                                5.70485e-1], dtype=np.longdouble)},
                2:{'a':np.array([8.830316038e2, 1.183989392e3,
                                4.473770672e2, 4.892542028e1,
                                1.000000000], dtype=np.longdouble),
                    'b':np.array([4.981972343e2, 1.020272984e3,
                                6.862151992e2, 1.728621255e2,
                                1.398575990e1, 2.138408204e-1], dtype=np.longdouble),
                    'c':np.array([-4.9141019880e-8, -7.2486358805e-6,
                                -7.4382915429e-4, -3.2856045308e-2,
                                -5.6853219702e-1, -1.9284139162,
                                1.0000000000], dtype=np.longdouble),
                    'd':np.array([-2.4570509894e-8, -3.6344227710e-6,
                                -3.7345152736e-4, -1.6589736860e-2,
                                -2.9154391835e-1, -1.1843742874e0,
                                7.0985168479e-1, -6.0197789199e-2], dtype=np.longdouble)},
                3:{'a':np.array([1.71446374704454e7, 3.88148302324068e7,
                                3.16743385304962e7, 1.14587609192151e7,
                                1.83696370756153e6, 1.14980998186874e5,
                                1.98276889924768e3, 1.00000000000000], dtype=np.longdouble),
                    'b':np.array([9.67282587452899e6, 2.87386436731785e7,
                                3.26070130734158e7, 1.77657027846367e7,
                                4.81648022267831e6, 6.13709569333207e5,
                                3.13595854332114e4, 4.35061725080755], dtype=np.longdouble),
                    'c':np.array([-4.46620341924942e-15, -1.58654991146236e-12,
                                -4.44467627042232e-10, -6.84738791621745e-8,
                                -6.64932238528105e-6, -3.69976170193942e-4,
                                -1.12295393687006e-2, -1.60926102124442e-1,
                                -8.52408612877447e-1, -7.45519953763928e-1,
                                2.98435207466372, 1.00000000000000], dtype=np.longdouble),
                    'd':np.array([-2.23310170962369e-15, -7.94193282071464e-13,
                                -2.22564376956228e-10, -3.43299431079845e-8,
                                -3.33919612678907e-6, -1.86432212187088e-4,
                                -5.69764436880529e-3, -8.34904593067194e-2,
                                -4.78770844009440e-1, -4.99759250374148e-1,
                                1.86795964993052, 4.16485970495288], dtype=np.double)}},
            0.5:{
                1:{'a':np.array([2.18168e1, 1.31693e1,
                                1.00000], dtype=np.longdouble),
                   'b':np.array([2.46180e1, 2.35546e1,
                                4.76290, 1.34481e-1], dtype=np.longdouble),
                   'c':np.array([4.73011e-2, 5.48433e-1,
                                1.00000], dtype=np.longdouble),
                   'd':np.array([7.09478e-2, 7.37041e-1,
                                3.82065e-1], dtype=np.longdouble)},
                2:{'a':np.array([7.940307136e2, 1.072518408e3,
                                4.106017002e2, 4.607473842e1,
                                1.000000000], dtype=np.longdouble),
                    'b':np.array([8.959677183e2, 1.526979592e3,
                                8.307577602e2, 1.638158630e2,
                                9.960923624, 1.047712331e-1], dtype=np.longdouble),
                    'c':np.array([7.265461948e-8, 1.678032858e-5,
                                1.365376899e-3, 4.647886226e-2,
                                5.231390123e-1, 1.567714263,
                                1.000000000], dtype=np.longdouble),
                    'd':np.array([1.089819298e-7, 2.503603684e-5,
                                2.017068914e-3, 6.719888328e-2,
                                7.001197631e-1, 1.309399040,
                                1.727377764e-1], dtype=np.longdouble)},
                3:{'a':np.array([5.75834152995465e6, 1.30964880355883e7,
                                1.07608632249013e7, 3.93536421893014e6,
                                6.42493233715640e5, 4.16031909245777e4,
                                7.77238678539648e2, 1.00000000000000], dtype=np.longdouble),
                    'b':np.array([6.49759261942269e6, 1.70750501625775e7,
                                1.69288134856160e7, 7.95192647756086e6,
                                1.83167424554505e6, 1.95155948326832e5,
                                8.17922106644547e3, 9.02129136642157e1], dtype=np.longdouble),
                    'c':np.array([4.8537838117341e-14, 1.64429113030738e-11,
                                3.76794942277806e-9, 4.69233883900644e-7,
                                3.40679845803144e-5, 1.32212995937796e-3,
                                2.60768398973913e-2, 2.48653216266227e-1,
                                1.08037861921488e0, 1.91247528779676e0,
                                1.00000000000000], dtype=np.longdouble),
                    'd':np.array([7.28067571760518e-14, 2.45745452167585e-11,
                                5.62152894375277e-9, 6.96888634549649e-7,
                                5.02360015186394e-5, 1.92040136756592e-3,
                                3.66887808002874e-2, 3.24095226486468e-1,
                                1.16434871200131, 1.34981244060549,
                                2.01311836975930e-1, -2.14562434782759e-2], dtype=np.longdouble)}},
            1.5:{
                1:{'a':np.array([1.35863e2, 4.92764e1,
                                1.00000], dtype=np.longdouble),
                    'b':np.array([1.02210e2, 5.50312e1,
                                4.23365e0], dtype=np.longdouble),
                    'c':np.array([1.54699e-1, 1.20037,
                                1.00000], dtype=np.longdouble),
                    'd':np.array([3.86765e-1, 6.08119e-1,
                                -1.65665e-1], dtype=np.longdouble)},
                2:{'a':np.array([9.895512903e2, 1.237156375e3,
                                4.413986183e2, 4.693212727e1,
                                1.000000000], dtype=np.longdouble),
                    'b':np.array([7.443927085e2, 1.062245497e3,
                                4.720721124e2, 7.386867306e1,
                                3.424526047, 2.473929073e-2], dtype=np.longdouble),
                    'c':np.array([6.7384341042e-8, 7.4281282702e-6,
                                4.6220789293e-4, 1.1905625478e-2,
                                1.3661062300e-1, 6.5500705397e-1,
                                1.0000000000], dtype=np.longdouble),
                    'd':np.array([1.6846085253e-7, 1.7531170088e-5,
                                1.0476768850e-3, 2.3334235654e-2, 
                                1.9947560547e-1, 4.7103657850e-1,
                                -1.7443752246e-2], dtype=np.longdouble)},
                3:{'a':np.array([4.32326386604283e4, 8.55472308218786e4,
                                5.95275291210962e4, 1.77294861572005e4,
                                2.21876607796460e3, 9.90562948053193e1,
                                1.00000000000000], dtype=np.longdouble),
                    'b':np.array([3.25218725353467e4, 7.01022511904373e4,
                                5.50859144223638e4, 1.95942074576400e4,
                                3.20803912586318e3, 2.20853967067789e2,
                                5.05580641737527, 1.99507945223266e-2], dtype=np.longdouble),
                    'c':np.array([2.80452693148553e-13, 8.60096863656367e-11,
                                1.62974620742993e-8, 1.63598843752050e-6,
                                9.12915407846722e-5, 2.62988766922117e-3, 
                                3.85682997219346e-2, 2.78383256609605e-1,
                                9.02250179334496e-1, 1.00000000000000], dtype=np.longdouble),
                    'd':np.array([7.01131732871184e-13, 2.10699282897576e-10,
                                3.94452010378723e-8, 3.84703231868724e-6,
                                2.04569943213216e-4, 5.31999109566385e-3,
                                6.39899717779153e-2, 3.14236143831882e-1,
                                4.70252591891375e-1, -2.15540156936373e-2,
                                2.34829436438087e-3], dtype=np.longdouble)}},
            2.5:{
                1:{'a':np.array([1.54674e2, 4.80784e1,
                                1.00000], dtype=np.longdouble),
                    'b':np.array([4.65428e1, 1.85625e1,
                                9.93679e-1], dtype=np.longdouble),
                    'c':np.array([5.69090e-1, 7.68654,
                                1.00000], dtype=np.longdouble),
                    'd':np.array([1.99168, -1.71711,
                                1.36953], dtype=np.longdouble)},
                2:{'a':np.array([1.178194436e4, 1.110612718e4,
                                2.722654825e3, 1.645171224e2,
                                1.000000000], dtype=np.longdouble),
                    'b':np.array([3.545200171e3, 3.655199255e3,
                                1.066529195e3, 9.326993632e1,
                                1.690677494], dtype=np.longdouble),
                    'c':np.array([1.4405190262e-6, 1.5534321883e-4,
                                6.9564011735e-3, 1.2618111665e-1, 
                                9.0276909572e-1, 1.9952283074,
                                1.0000000000], dtype=np.longdouble),
                    'd':np.array([5.0418165971e-6, 4.7113349177e-4,
                                1.7503664846e-2, 1.8378232714e-1,
                                2.9430307063e-1, 3.2980790411e-2], dtype=np.longdouble)},
                3:{'a':np.array([6.61606300631656e4, 1.20132462801652e5,
                                7.67255995316812e4, 2.10427138842443e4,
                                2.44325236813275e3, 1.02589947781696e2,
                                1.00000000000000], dtype=np.longdouble),
                    'b':np.array([1.99078071053871e4, 3.79076097261066e4,
                                2.60117136841197e4, 7.97584657659364e3,
                                1.10886130159658e3, 6.35483623268093e1,
                                1.16951072617142, 3.31482978240026e-3], dtype=np.longdouble),
                    'c':np.array([8.42667076131315e-12, 2.31618876821567e-9,
                                3.54323824923987e-7, 2.77981736000034e-5,
                                1.14008027400645e-3, 2.32779790773633e-2,
                                2.39564845938301e-1, 1.24415366126179,
                                3.18831203950106, 3.42040216997894,
                                1.00000000000000], dtype=np.longdouble),
                    'd':np.array([2.94933476646033e-11, 7.68215783076936e-9,
                                1.12919616415947e-6, 8.09451165406274e-5,
                                2.81111224925648e-3, 3.99937801931919e-2,
                                2.27132567866839e-1, 5.31886045222680e-1,
                                3.70866321410385e-1, 2.27326643192516], dtype=np.longdouble)}}
}


#Inverse Fermi Integral Coeffs. Accessed by icoeffs[n][order][letter] with
#order in (1,2)
#n in (-0.5, 0.5, 1.5, 2.5)
#letter in ('a', 'b', 'c', 'd')'
icoeffs = {-0.5:{1:{'a':np.array([7.8516685e2, -1.4034065e2,
                                1.3257418e1, 1.0000000], dtype=np.longdouble),
                    'b':np.array([1.3917278e3, -8.0463066e2,
                                1.5854806e2, -1.0640712e1], dtype=np.longdouble),
                    'c':np.array([8.9742174e-3, -1.0604768e-1,
                                1.0000000], dtype=np.longdouble),
                    'd':np.array([3.5898124e-2, -4.2520975e-1,
                                3.6612154], dtype=np.longdouble)},
                2:{'a':np.array([-1.570044577033e4, 1.001958278442e4,
                                -2.805343454951e3, 4.121170498099e2,
                                -3.174780572961e1, 1.000000000000], dtype=np.longdouble),
                    'b':np.array([-2.782831558471e4, 2.886114034012e4,
                                -1.274243093149e4, 3.063252215963e3,
                                -4.225615045074e2, 3.168918168284e1,
                                -1.008561571363], dtype=np.longdouble),
                    'c':np.array([2.20677916003e-8, -1.437701234283e-6,
                                6.103116850636e-5, -1.169411057416e-3,
                                1.814141021608e-2, -9.588603457639e-2,
                                1.000000000000], dtype=np.longdouble),
                    'd':np.array([8.827116613576e-8, -5.750804196059e-6,
                                2.429627688357e-4, -4.601959491394e-3,
                                6.932122275919e-2, -3.217372489776e-1,
                                3.124344749296], dtype=np.longdouble)}
                },
            0.5:{1:{'a':np.array([4.4593646e1, 1.1288764e1,
                                1.0000000], dtype=np.longdouble),
                    'b':np.array([3.9519346e1, -5.7517464,
                                2.6594291e-1], dtype=np.longdouble),
                    'c':np.array([3.4873722e1, -2.6922515e1,
                                1.0000000], dtype=np.longdouble),
                    'd':np.array([2.6612832e1, -2.0452930e1,
                                1.1808945e1], dtype=np.longdouble)},
                2:{'a':np.array([1.999266880833e4, 5.702479099336e3,
                                6.610132843877e2, 3.818838129486e1,
                                1.000000000000], dtype=np.longdouble),
                    'b':np.array([1.771804140488e4, -2.014785161019e3,
                                9.130355392717e1, -1.670718177489], dtype=np.longdouble),
                    'c':np.array([-1.277060388085e-2, 7.187946804945e-2,
                                -4.262314235106e-1, 4.997559426872e-1,
                                -1.285579118012, -3.930805454272e-1,
                                1.000000000000], dtype=np.longdouble),
                    'd':np.array([-9.745794806288e-3, 5.485432756838e-2,
                                -3.299466243260e-1, 4.077841975923e-1,
                                -1.145531476975, -6.067091689181e-2], dtype=np.longdouble)}
                },
            1.5:{1:{'a':np.array([3.5954549e1, 1.3908910e1, 
                                1.0000000], dtype=np.longdouble),
                    'b':np.array([4.7795853e1, 1.2133628e1,
                                -2.3975074e-1], dtype=np.longdouble),
                    'c':np.array([-9.8934493e-1, 9.0731169e-2, 
                                1.0000000], dtype=np.longdouble),
                    'd':np.array([-6.8577484e-1, 6.3338994e-2,
                                -1.1635840e-1], dtype=np.longdouble)},
                2:{'a':np.array([1.715627994191e2, 1.125926232897e2,
                                2.056296753055e1, 1.000000000000], dtype=np.longdouble),
                    'b':np.array([2.280653583157e2, 1.193456203021e2,
                                1.167743113540e1, -3.226808804038e-1,
                                3.519268762788e-3], dtype=np.longdouble),
                    'c':np.array([-6.321828169799e-3, -2.183147266896e-2,
                                -1.057562799320e-1, -4.657944387545e-1,
                                -5.951932864088e-1, 3.684471177100e-1,
                                1.000000000000], dtype=np.longdouble),
                    'd':np.array([-4.381942605018e-3, -1.513236504100e-2,
                                -7.850001283886e-2, -3.407561772612e-1,
                                -5.074812565486e-1, -1.387107009074e-1], dtype=np.longdouble)}
                },
            2.5:{1:{'a':np.array([1.5331469e1, 1.0000000], dtype=np.longdouble),
                    'b':np.array([5.0951752e1, 1.9691913,
                                -2.7251177e-2], dtype=np.longdouble),
                    'c':np.array([-5.1891788e-1, -9.1723019e-3,
                                1.0000000], dtype=np.longdouble),
                    'd':np.array([-3.6278896e-1, -6.1502672e-3,
                                -3.3673540e-2], dtype=np.longdouble)},
                2:{'a':np.array([2.138969250409e2, 3.539903493971e1,
                                1.000000000000], dtype=np.longdouble),
                    'b':np.array([7.108545512710e2, 9.873746988121e1,
                                1.067755522895, -1.182798726503e-2], dtype=np.longdouble),
                    'c':np.array([-3.312041011227e-2, 1.315763372315e-1,
                                -4.820942898296e-1, 5.099038074944e-1,
                                5.495613498630e-1, -1.498867562255,
                                1.000000000000], dtype=np.longdouble),
                    'd':np.array([-2.315515517515e-2, 9.198776585252e-2,
                                -3.835879295548e-1, 5.415026856351e-1,
                                -3.847241692193e-1, 3.739781456585e-2,
                                -3.008504449098e-2], dtype=np.longdouble)}
                }
}


def fint(x, n, order=3):
    """Evaluates a Pade fit to the fermi integral of order n.
    
    Parameters
    ----------
    x : float
        Point at which to evaluate function.
    n : {-0.5, 0.5, 1.5, 2.5}
        Which fermi integral to evaluate.
    order : optional, {1, 2, 3}
    
    Returns
    -------
    out : float
        Function value.
    """
    out = 0.0
    denom = 0.0
    if x <2.0:
        x = np.exp(x)
        out = coeffs[n][order]['a'][-1]
        denom = coeffs[n][order]['b'][-1]
        for i in range(coeffs[n][order]['a'].shape[0]-2, -1, -1):
            out = coeffs[n][order]['a'][i] + x*out
        for i in range(coeffs[n][order]['b'].shape[0]-2, -1, -1):
            denom = coeffs[n][order]['b'][i] + x*denom
        out = x*out/denom
    else:
        x = 1/(x**2)
        out = coeffs[n][order]['c'][-1]
        denom = coeffs[n][order]['d'][-1]
        for i in range(coeffs[n][order]['c'].shape[0]-2, -1, -1):
            out = coeffs[n][order]['c'][i] + x*out
        for i in range(coeffs[n][order]['d'].shape[0]-2, -1, -1):
            denom = coeffs[n][order]['d'][i] + x*denom
        out = np.power(x, -(n+1)/2)*out/denom
    return out


def ifint(f, n, order=2):
    """Evaluates a Pade fit to the inverse of the fermi integral of order n.

    Parameters
    ----------
    f : float
        Point at which to evaluate the inverse function.
    n : {-0.5, 0.5, 1.5, 2.5}
        Which fermi integral to get the inverse of.
    order : optional, {1,2,3}

    Returns
    -------
    out : float
        Inverse Function Value.
    """
    out = 0.0
    denom = 0.0
    if f <4.0:
        out = coeffs[n][order]['a'][-1]
        denom = coeffs[n][order]['b'][-1]
        for i in range(coeffs[n][order]['a'].shape[0]-2, -1, -1):
            out = coeffs[n][order]['a'][i] + f*out
        for i in range(coeffs[n][order]['b'].shape[0]-2, -1, -1):
            denom = coeffs[n][order]['b'][i] + f*denom
        out = np.log(f*out/denom)
    else:
        f = np.power(f, -1.0/(1+n))
        out = coeffs[n][order]['c'][-1]
        denom = coeffs[n][order]['d'][-1]
        for i in range(coeffs[n][order]['c'].shape[0]-2, -1, -1):
            out = coeffs[n][order]['c'][i] + f*out
        for i in range(coeffs[n][order]['d'].shape[0]-2, -1, -1):
            denom = coeffs[n][order]['d'][i] + f*denom
        out = out/denom/f
    return out


def fint_arr(x, n, order=3):
    """Evaluates a Pade fit to the fermi integral of order n for an array of
    points.
    
    Parameters
    ----------
    x : float
        Point at which to evaluate function.
    n : {-0.5, 0.5, 1.5, 2.5}
        Which fermi integral to evaluate.
    order : optional, {1, 2, 3}
    
    Returns
    -------
    out : float
        Function value.
    """
    out = np.zeros_like(x)
    denom = np.zeros_like(x)
    c1 = x < 2.0
    c2 = x >= 2.0
    x[c1] = np.exp(x[c1])
    x[c2] = 1/x[c2]/x[c2]
    out[c1] = coeffs[n][order]['a'][-1]
    denom[c1] = coeffs[n][order]['b'][-1]
    out[c2] = coeffs[n][order]['c'][-1]
    denom[c2] = coeffs[n][order]['d'][-1]
    for i in range(coeffs[n][order]['a'].shape[0]-2, -1, -1):
        out[c1] = coeffs[n][order]['a'][i] + x[c1]*out[c1]
    for i in range(coeffs[n][order]['b'].shape[0]-2, -1, -1):
        denom[c1] = coeffs[n][order]['b'][i] + x[c1]*denom[c1]
    for i in range(coeffs[n][order]['c'].shape[0]-2, -1, -1):
        out[c2] = coeffs[n][order]['c'][i] + x[c2]*out[c2]
    for i in range(coeffs[n][order]['d'].shape[0]-2, -1, -1):
        denom[c2] = coeffs[n][order]['d'][i] + x[c2]*denom[c2]
    out[c1] = x[c1]*out[c1]/denom[c1]
    out[c2] = np.power(x[c2], -(n+1)/2)*out[c2]/denom[c2]
    return out


def ifint_arr(f, n, order=2):
    """Evaluates a Pade fit to the inverse of the fermi integral of order n
    for an array of points.

    Parameters
    ----------
    f : float
        Point at which to evaluate the inverse function.
    n : {-0.5, 0.5, 1.5, 2.5}
        Which fermi integral to get the inverse of.
    order : optional, {1,2,3}

    Returns
    -------
    out : float
        Inverse Function Value.
    """
    out = np.zeros_like(f)
    denom = np.zeros_like(f)
    c1 = f<4.0
    c2 = f>= 4.0
    f[c2] = np.power(f[c2], -1.0/(1+n))
    #horners method
    out[c1] = f[c1]*icoeffs[n][order]['a'][-1]
    out[c2] = f[c2]*icoeffs[n][order]['c'][-1]
    denom[c1] = f[c1]*icoeffs[n][order]['b'][-1]
    denom[c2] = f[c2]*icoeffs[n][order]['d'][-1]
    for i in range(icoeffs[n][order]['a'].shape[0]-2, -1, -1):
        out[c1] = icoeffs[n][order]['a'][i] + f[c1]*out[c1]
    for i in range(icoeffs[n][order]['b'].shape[0]-2, -1, -1):
        denom[c1] = icoeffs[n][order]['b'][i] + f[c1]*denom[c1]
    for i in range(icoeffs[n][order]['c'].shape[0]-2, -1, -1):
        out[c2] = icoeffs[n][order]['c'][i] + f[c2]*out[c2]
    for i in range(icoeffs[n][order]['d'].shape[0]-2, -1, -1):
        denom[c2] = icoeffs[n][order]['d'][i] + f[c2]*denom[c2]
    #import pdb
    #pdb.set_trace()
    out[c1] = np.log(f[c1]*out[c1]/denom[c1])
    out[c2] = out[c2]/denom[c2]/f[c2]
    return out