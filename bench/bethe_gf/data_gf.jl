# QInchworm.jl
#
# Copyright (C) 2021-2023 I. Krivenko, H. U. R. Strand and J. Kleinhenz
#
# QInchworm.jl is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# QInchworm.jl is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# QInchworm.jl. If not, see <http://www.gnu.org/licenses/.
#
# Author: Hugo U. R. Strand

function get_g_nca()
g = -im *[-0.443594  , -0.42008018, -0.39873717, -0.37932407, -0.36163056,
       -0.34547259, -0.33068882, -0.31713748, -0.30469376, -0.29324757,
       -0.28270159, -0.27296968, -0.2639754 , -0.2556508 , -0.2479354 ,
       -0.24077525, -0.23412216, -0.22793298, -0.22216907, -0.21679569,
       -0.21178164, -0.20709881, -0.20272185, -0.19862786, -0.19479614,
       -0.19120793, -0.18784624, -0.18469562, -0.18174206, -0.17897279,
       -0.17637618, -0.17394164, -0.1716595 , -0.16952092, -0.16751785,
       -0.1656429 , -0.16388932, -0.16225094, -0.1607221 , -0.15929763,
       -0.15797279, -0.15674325, -0.15560503, -0.15455452, -0.15358841,
       -0.15270368, -0.1518976 , -0.15116766, -0.15051161, -0.14992742,
       -0.14941327, -0.14896752, -0.14858873, -0.14827563, -0.14802713,
       -0.14784228, -0.14772029, -0.14766055, -0.14766254, -0.14772594,
       -0.14785052, -0.14803622, -0.14828309, -0.14859134, -0.1489613 ,
       -0.14939342, -0.14988832, -0.15044674, -0.15106955, -0.15175779,
       -0.15251263, -0.15333539, -0.15422758, -0.15519084, -0.15622701,
       -0.15733808, -0.15852627, -0.15979397, -0.16114381, -0.16257861,
       -0.16410147, -0.16571573, -0.16742498, -0.16923314, -0.17114442,
       -0.17316337, -0.17529492, -0.17754436, -0.17991741, -0.18242027,
       -0.18505961, -0.18784262, -0.1907771 , -0.19387146, -0.19713481,
       -0.20057699, -0.20420866, -0.20804139, -0.21208769, -0.21636116,
       -0.22087657, -0.22564996, -0.2306988 , -0.2360421 , -0.24170058,
       -0.24769688, -0.2540557 , -0.26080406, -0.26797153, -0.27559054,
       -0.28369666, -0.29232897, -0.30153045, -0.31134845, -0.32183518,
       -0.33304828, -0.34505148, -0.35791536, -0.37171816, -0.38654675,
       -0.40249773, -0.41967865, -0.43820948, -0.45822421, -0.47987272,
       -0.50332296, -0.52876343, -0.556406  ]
return g
end
function get_g_oca()
g = -im *[-0.44950003, -0.43140822, -0.41472244, -0.39930261, -0.38502587,
       -0.37178412, -0.359482  , -0.3480351 , -0.3373685 , -0.32741554,
       -0.31811674, -0.30941886, -0.3012742 , -0.29363986, -0.2864772 ,
       -0.27975135, -0.27343075, -0.26748681, -0.26189357, -0.25662742,
       -0.25166683, -0.24699218, -0.24258553, -0.23843046, -0.23451193,
       -0.23081613, -0.22733039, -0.22404307, -0.22094343, -0.2180216 ,
       -0.21526848, -0.21267566, -0.21023541, -0.20794055, -0.2057845 ,
       -0.20376113, -0.20186482, -0.20009035, -0.19843292, -0.19688807,
       -0.19545171, -0.19412007, -0.19288966, -0.19175728, -0.19072   ,
       -0.18977512, -0.18892019, -0.18815296, -0.1874714 , -0.18687368,
       -0.18635814, -0.18592331, -0.1855679 , -0.18529077, -0.18509095,
       -0.18496759, -0.18492004, -0.18494776, -0.18505035, -0.18522755,
       -0.18547926, -0.18580548, -0.18620636, -0.18668218, -0.18723334,
       -0.18786038, -0.18856398, -0.18934494, -0.19020419, -0.19114281,
       -0.19216202, -0.19326316, -0.19444775, -0.19571741, -0.19707396,
       -0.19851936, -0.20005573, -0.20168535, -0.2034107 , -0.20523444,
       -0.20715939, -0.2091886 , -0.21132533, -0.21357303, -0.21593542,
       -0.21841642, -0.22102024, -0.22375133, -0.22661443, -0.22961458,
       -0.23275715, -0.2360478 , -0.2394926 , -0.24309793, -0.24687062,
       -0.25081789, -0.25494741, -0.25926734, -0.26378632, -0.26851355,
       -0.2734588 , -0.27863243, -0.28404547, -0.28970965, -0.29563744,
       -0.30184209, -0.30833773, -0.31513938, -0.32226305, -0.3297258 ,
       -0.33754581, -0.34574249, -0.35433653, -0.36335004, -0.37280665,
       -0.38273162, -0.39315198, -0.40409667, -0.41559672, -0.42768539,
       -0.44039841, -0.45377416, -0.46785395, -0.48268226, -0.49830705,
       -0.51478014, -0.53215755, -0.55049997]
return g
end
function get_g_tca()
g = -im *[-0.45314301, -0.43663259, -0.42118899, -0.40674017, -0.39321816,
       -0.3805591 , -0.36870324, -0.35759483, -0.34718198, -0.3374165 ,
       -0.32825372, -0.31965223, -0.31157376, -0.3039829 , -0.29684695,
       -0.2901357 , -0.28382128, -0.27787794, -0.27228192, -0.26701131,
       -0.26204585, -0.25736688, -0.25295714, -0.24880071, -0.24488287,
       -0.24119004, -0.23770965, -0.2344301 , -0.23134067, -0.22843142,
       -0.2256932 , -0.2231175 , -0.2206965 , -0.21842292, -0.21629005,
       -0.21429168, -0.21242207, -0.2106759 , -0.20904829, -0.20753469,
       -0.20613093, -0.20483315, -0.2036378 , -0.20254163, -0.20154163,
       -0.20063507, -0.19981942, -0.19909242, -0.19845197, -0.19789621,
       -0.19742346, -0.19703219, -0.1967211 , -0.19648899, -0.19633487,
       -0.19625787, -0.19625729, -0.19633255, -0.19648324, -0.19670904,
       -0.19700981, -0.19738552, -0.19783626, -0.19836225, -0.19896386,
       -0.19964156, -0.20039595, -0.20122778, -0.2021379 , -0.20312731,
       -0.20419712, -0.20534861, -0.20658315, -0.20790229, -0.2093077 ,
       -0.21080121, -0.21238481, -0.21406062, -0.21583096, -0.21769828,
       -0.21966526, -0.2217347 , -0.22390966, -0.22619334, -0.22858919,
       -0.23110087, -0.23373228, -0.23648755, -0.23937106, -0.2423875 ,
       -0.24554179, -0.24883919, -0.25228527, -0.25588591, -0.25964737,
       -0.26357627, -0.26767962, -0.27196486, -0.27643984, -0.28111291,
       -0.28599289, -0.29108912, -0.29641149, -0.30197046, -0.30777711,
       -0.31384316, -0.320181  , -0.32680374, -0.33372524, -0.34096011,
       -0.34852383, -0.3564327 , -0.36470393, -0.37335562, -0.38240685,
       -0.39187768, -0.40178912, -0.41216324, -0.42302306, -0.43439263,
       -0.44629692, -0.45876183, -0.47181405, -0.48548098, -0.49979054,
       -0.51477096, -0.53045048, -0.54685699]
return g
end
function get_P_nca()
P0 = [-1.        , -0.95576893, -0.9144512 , -0.87581573, -0.83965236,
       -0.80576973, -0.77399345, -0.74416436, -0.71613713, -0.68977884,
       -0.66496787, -0.64159278, -0.61955139, -0.59874992, -0.57910219,
       -0.56052897, -0.54295734, -0.52632012, -0.5105554 , -0.49560603,
       -0.48141924, -0.46794626, -0.45514198, -0.44296466, -0.43137561,
       -0.42033897, -0.40982148, -0.39979225, -0.39022259, -0.38108582,
       -0.37235712, -0.36401338, -0.35603308, -0.34839615, -0.34108386,
       -0.33407876, -0.32736451, -0.32092587, -0.31474859, -0.30881931,
       -0.30312555, -0.2976556 , -0.29239852, -0.287344  , -0.28248242,
       -0.27780472, -0.2733024 , -0.26896747, -0.26479242, -0.26077021,
       -0.25689418, -0.25315811, -0.24955609, -0.2460826 , -0.24273242,
       -0.23950064, -0.23638262, -0.233374  , -0.23047067, -0.22766874,
       -0.22496456, -0.22235468, -0.21983584, -0.21740498, -0.21505922,
       -0.21279581, -0.21061221, -0.208506  , -0.20647491, -0.20451679,
       -0.20262964, -0.20081158, -0.19906085, -0.19737579, -0.19575485,
       -0.19419661, -0.19269972, -0.19126294, -0.18988512, -0.18856522,
       -0.18730227, -0.1860954 , -0.1849438 , -0.18384677, -0.1828037 ,
       -0.18181403, -0.18087731, -0.17999315, -0.17916126, -0.17838143,
       -0.17765351, -0.17697745, -0.17635329, -0.17578114, -0.17526122,
       -0.1747938 , -0.1743793 , -0.17401817, -0.17371102, -0.17345853,
       -0.17326148, -0.17312078, -0.17303747, -0.17301268, -0.17304769,
       -0.17314392, -0.17330294, -0.17352646, -0.17381636, -0.17417471,
       -0.17460376, -0.17510596, -0.17568398, -0.17634071, -0.17707932,
       -0.17790323, -0.17881614, -0.17982208, -0.18092544, -0.18213093,
       -0.18344372, -0.18486937, -0.18641394, -0.188084  , -0.18988667,
       -0.19182973, -0.19392158, -0.1961714 ]
P1 = [-1.        , -0.95599319, -0.91529125, -0.87758725, -0.84260668,
       -0.81010364, -0.77985761, -0.75167055, -0.72536445, -0.70077901,
       -0.6777698 , -0.65620644, -0.63597114, -0.61695737, -0.5990686 ,
       -0.58221732, -0.56632408, -0.55131666, -0.53712934, -0.52370225,
       -0.51098077, -0.49891504, -0.48745946, -0.47657231, -0.46621537,
       -0.45635355, -0.44695468, -0.43798914, -0.42942972, -0.42125132,
       -0.4134308 , -0.40594683, -0.39877967, -0.39191108, -0.38532415,
       -0.37900325, -0.37293384, -0.36710245, -0.36149656, -0.35610451,
       -0.35091546, -0.3459193 , -0.34110662, -0.33646861, -0.33199708,
       -0.32768434, -0.32352322, -0.319507  , -0.31562938, -0.31188448,
       -0.30826675, -0.30477101, -0.30139239, -0.29812631, -0.29496848,
       -0.29191484, -0.28896159, -0.28610516, -0.28334217, -0.28066947,
       -0.27808405, -0.27558312, -0.27316403, -0.27082429, -0.26856154,
       -0.26637359, -0.26425836, -0.26221389, -0.26023835, -0.25833001,
       -0.25648726, -0.25470859, -0.25299257, -0.25133787, -0.24974327,
       -0.24820762, -0.24672984, -0.24530895, -0.24394404, -0.24263429,
       -0.24137893, -0.24017727, -0.23902871, -0.23793268, -0.23688873,
       -0.23589643, -0.23495544, -0.2340655 , -0.23322639, -0.23243797,
       -0.23170017, -0.23101298, -0.23037648, -0.22979079, -0.22925613,
       -0.22877277, -0.22834108, -0.22796148, -0.2276345 , -0.22736072,
       -0.22714085, -0.22697564, -0.22686596, -0.2268128 , -0.2268172 ,
       -0.22688035, -0.22700354, -0.22718818, -0.22743581, -0.2277481 ,
       -0.22812686, -0.22857406, -0.22909183, -0.22968247, -0.23034846,
       -0.23109248, -0.23191743, -0.23282641, -0.23382278, -0.23491017,
       -0.23609245, -0.23737384, -0.23875885, -0.24025236, -0.2418596 ,
       -0.24358624, -0.24543838, -0.2474226 ]
P2 = [-1.        , -0.95599319, -0.91529125, -0.87758725, -0.84260668,
       -0.81010364, -0.77985761, -0.75167055, -0.72536445, -0.70077901,
       -0.6777698 , -0.65620644, -0.63597114, -0.61695737, -0.5990686 ,
       -0.58221732, -0.56632408, -0.55131666, -0.53712934, -0.52370225,
       -0.51098077, -0.49891504, -0.48745946, -0.47657231, -0.46621537,
       -0.45635355, -0.44695468, -0.43798914, -0.42942972, -0.42125132,
       -0.4134308 , -0.40594683, -0.39877967, -0.39191108, -0.38532415,
       -0.37900325, -0.37293384, -0.36710245, -0.36149656, -0.35610451,
       -0.35091546, -0.3459193 , -0.34110662, -0.33646861, -0.33199708,
       -0.32768434, -0.32352322, -0.319507  , -0.31562938, -0.31188448,
       -0.30826675, -0.30477101, -0.30139239, -0.29812631, -0.29496848,
       -0.29191484, -0.28896159, -0.28610516, -0.28334217, -0.28066947,
       -0.27808405, -0.27558312, -0.27316403, -0.27082429, -0.26856154,
       -0.26637359, -0.26425836, -0.26221389, -0.26023835, -0.25833001,
       -0.25648726, -0.25470859, -0.25299257, -0.25133787, -0.24974327,
       -0.24820762, -0.24672984, -0.24530895, -0.24394404, -0.24263429,
       -0.24137893, -0.24017727, -0.23902871, -0.23793268, -0.23688873,
       -0.23589643, -0.23495544, -0.2340655 , -0.23322639, -0.23243797,
       -0.23170017, -0.23101298, -0.23037648, -0.22979079, -0.22925613,
       -0.22877277, -0.22834108, -0.22796148, -0.2276345 , -0.22736072,
       -0.22714085, -0.22697564, -0.22686596, -0.2268128 , -0.2268172 ,
       -0.22688035, -0.22700354, -0.22718818, -0.22743581, -0.2277481 ,
       -0.22812686, -0.22857406, -0.22909183, -0.22968247, -0.23034846,
       -0.23109248, -0.23191743, -0.23282641, -0.23382278, -0.23491017,
       -0.23609245, -0.23737384, -0.23875885, -0.24025236, -0.2418596 ,
       -0.24358624, -0.24543838, -0.2474226 ]
P3 = [-1.        , -0.95621749, -0.91613181, -0.8793612 , -0.84556807,
       -0.81445347, -0.78575226, -0.759229  , -0.73467427, -0.71190162,
       -0.69074474, -0.67105515, -0.65270006, -0.63556059, -0.61953014,
       -0.60451303, -0.59042324, -0.57718334, -0.56472357, -0.55298096,
       -0.54189862, -0.53142507, -0.52151369, -0.51212221, -0.50321222,
       -0.49474882, -0.48670023, -0.47903747, -0.47173414, -0.46476607,
       -0.45811119, -0.45174928, -0.44566179, -0.4398317 , -0.43424339,
       -0.42888247, -0.42373568, -0.41879082, -0.41403662, -0.40946266,
       -0.40505932, -0.40081767, -0.39672945, -0.392787  , -0.38898321,
       -0.38531145, -0.38176557, -0.37833986, -0.37502896, -0.37182789,
       -0.36873203, -0.36573701, -0.36283879, -0.36003356, -0.35731779,
       -0.35468813, -0.35214148, -0.34967491, -0.34728568, -0.34497121,
       -0.34272909, -0.34055705, -0.33845296, -0.33641482, -0.33444075,
       -0.33252899, -0.33067786, -0.32888583, -0.32715141, -0.32547324,
       -0.32385004, -0.32228059, -0.32076376, -0.3192985 , -0.31788383,
       -0.31651881, -0.31520259, -0.31393439, -0.31271345, -0.3115391 ,
       -0.31041072, -0.30932772, -0.3082896 , -0.30729588, -0.30634614,
       -0.30544   , -0.30457714, -0.30375727, -0.30298016, -0.30224562,
       -0.30155349, -0.30090369, -0.30029614, -0.29973084, -0.29920782,
       -0.29872715, -0.29828896, -0.2978934 , -0.2975407 , -0.29723111,
       -0.29696494, -0.29674255, -0.29656434, -0.29643079, -0.2963424 ,
       -0.29629974, -0.29630344, -0.2963542 , -0.29645275, -0.29659992,
       -0.29679659, -0.29704371, -0.29734231, -0.29769351, -0.29809848,
       -0.29855852, -0.29907499, -0.29964935, -0.30028318, -0.30097817,
       -0.30173609, -0.30255889, -0.30344861, -0.30440745, -0.30543775,
       -0.30654202, -0.30772295, -0.3089834 ]

return P0, P1, P2, P3

end
function get_P_oca()
P0 = [-1.        , -0.95201873, -0.90728941, -0.86554819, -0.82655597,
       -0.79009574, -0.75597032, -0.72400027, -0.69402216, -0.66588687,
       -0.63945824, -0.6146118 , -0.59123358, -0.56921916, -0.54847275,
       -0.52890637, -0.51043918, -0.49299674, -0.47651052, -0.46091733,
       -0.44615885, -0.43218121, -0.41893463, -0.40637301, -0.3944537 ,
       -0.38313715, -0.37238666, -0.3621682 , -0.35245013, -0.34320304,
       -0.33439957, -0.32601425, -0.31802334, -0.31040472, -0.30313777,
       -0.2962032 , -0.28958303, -0.28326045, -0.27721973, -0.27144615,
       -0.26592596, -0.26064624, -0.25559491, -0.25076064, -0.2461328 ,
       -0.24170141, -0.23745712, -0.23339114, -0.22949519, -0.22576153,
       -0.22218285, -0.21875228, -0.21546339, -0.21231009, -0.20928667,
       -0.20638775, -0.20360828, -0.20094349, -0.1983889 , -0.19594029,
       -0.19359368, -0.19134535, -0.18919178, -0.18712967, -0.18515591,
       -0.18326758, -0.18146194, -0.17973644, -0.17808866, -0.17651635,
       -0.17501741, -0.17358988, -0.17223193, -0.17094187, -0.16971812,
       -0.16855923, -0.16746387, -0.16643082, -0.16545898, -0.16454734,
       -0.163695  , -0.16290117, -0.16216516, -0.16148638, -0.16086435,
       -0.16029866, -0.15978903, -0.15933526, -0.15893726, -0.15859505,
       -0.15830872, -0.15807851, -0.15790472, -0.1577878 , -0.15772829,
       -0.15772685, -0.15778427, -0.15790145, -0.15807944, -0.15831942,
       -0.1586227 , -0.15899078, -0.15942527, -0.15992801, -0.16050097,
       -0.16114634, -0.16186651, -0.16266409, -0.16354193, -0.16450313,
       -0.16555106, -0.16668938, -0.16792209, -0.16925351, -0.17068833,
       -0.17223166, -0.17388903, -0.17566645, -0.17757046, -0.17960815,
       -0.18178723, -0.18411609, -0.18660385, -0.18926045, -0.19209671,
       -0.19512444, -0.19835653, -0.20180704]
P1 = [-1.        , -0.95224213, -0.90812316, -0.8673003 , -0.82946814,
       -0.79435418, -0.76171479, -0.73133188, -0.70300984, -0.67657299,
       -0.65186326, -0.62873824, -0.60706941, -0.58674061, -0.5676467 ,
       -0.54969238, -0.53279115, -0.51686435, -0.5018404 , -0.48765404,
       -0.47424571, -0.46156096, -0.44954999, -0.43816713, -0.42737052,
       -0.41712168, -0.40738525, -0.39812865, -0.38932189, -0.38093725,
       -0.37294917, -0.36533399, -0.35806982, -0.35113638, -0.34451487,
       -0.33818781, -0.33213901, -0.32635336, -0.32081685, -0.31551639,
       -0.31043981, -0.30557572, -0.30091354, -0.29644333, -0.29215583,
       -0.28804238, -0.28409486, -0.28030567, -0.27666769, -0.27317425,
       -0.26981908, -0.26659631, -0.26350043, -0.26052625, -0.25766891,
       -0.25492384, -0.25228674, -0.24975358, -0.24732056, -0.2449841 ,
       -0.24274087, -0.24058769, -0.23852162, -0.23653987, -0.23463983,
       -0.23281904, -0.23107521, -0.22940618, -0.22780993, -0.22628458,
       -0.22482837, -0.22343965, -0.22211691, -0.22085872, -0.21966377,
       -0.21853086, -0.21745889, -0.21644682, -0.21549375, -0.21459885,
       -0.21376138, -0.21298067, -0.21225618, -0.2115874 , -0.21097393,
       -0.21041547, -0.20991178, -0.2094627 , -0.20906816, -0.20872818,
       -0.20844284, -0.20821235, -0.20803696, -0.20791703, -0.20785301,
       -0.20784545, -0.20789497, -0.20800233, -0.20816835, -0.20839399,
       -0.20868031, -0.20902847, -0.20943979, -0.20991568, -0.21045772,
       -0.2110676 , -0.21174718, -0.21249848, -0.21332368, -0.21422517,
       -0.21520548, -0.2162674 , -0.21741391, -0.21864823, -0.21997384,
       -0.22139447, -0.22291416, -0.22453727, -0.22626847, -0.22811282,
       -0.23007577, -0.23216317, -0.23438135, -0.23673715, -0.23923794,
       -0.24189166, -0.24470691, -0.24769299]
P2 = [-1.        , -0.95224213, -0.90812316, -0.8673003 , -0.82946814,
       -0.79435418, -0.76171479, -0.73133188, -0.70300984, -0.67657299,
       -0.65186326, -0.62873824, -0.60706941, -0.58674061, -0.5676467 ,
       -0.54969238, -0.53279115, -0.51686435, -0.5018404 , -0.48765404,
       -0.47424571, -0.46156096, -0.44954999, -0.43816713, -0.42737052,
       -0.41712168, -0.40738525, -0.39812865, -0.38932189, -0.38093725,
       -0.37294917, -0.36533399, -0.35806982, -0.35113638, -0.34451487,
       -0.33818781, -0.33213901, -0.32635336, -0.32081685, -0.31551639,
       -0.31043981, -0.30557572, -0.30091354, -0.29644333, -0.29215583,
       -0.28804238, -0.28409486, -0.28030567, -0.27666769, -0.27317425,
       -0.26981908, -0.26659631, -0.26350043, -0.26052625, -0.25766891,
       -0.25492384, -0.25228674, -0.24975358, -0.24732056, -0.2449841 ,
       -0.24274087, -0.24058769, -0.23852162, -0.23653987, -0.23463983,
       -0.23281904, -0.23107521, -0.22940618, -0.22780993, -0.22628458,
       -0.22482837, -0.22343965, -0.22211691, -0.22085872, -0.21966377,
       -0.21853086, -0.21745889, -0.21644682, -0.21549375, -0.21459885,
       -0.21376138, -0.21298067, -0.21225618, -0.2115874 , -0.21097393,
       -0.21041547, -0.20991178, -0.2094627 , -0.20906816, -0.20872818,
       -0.20844284, -0.20821235, -0.20803696, -0.20791703, -0.20785301,
       -0.20784545, -0.20789497, -0.20800233, -0.20816835, -0.20839399,
       -0.20868031, -0.20902847, -0.20943979, -0.20991568, -0.21045772,
       -0.2110676 , -0.21174718, -0.21249848, -0.21332368, -0.21422517,
       -0.21520548, -0.2162674 , -0.21741391, -0.21864823, -0.21997384,
       -0.22139447, -0.22291416, -0.22453727, -0.22626847, -0.22811282,
       -0.23007577, -0.23216317, -0.23438135, -0.23673715, -0.23923794,
       -0.24189166, -0.24470691, -0.24769299]
P3 = [-1.        , -0.95246558, -0.90895768, -0.86905594, -0.83239057,
       -0.79863556, -0.76750289, -0.73873768, -0.71211381, -0.68743041,
       -0.66450861, -0.64318887, -0.62332865, -0.60480033, -0.58748947,
       -0.57129329, -0.5561193 , -0.54188414, -0.52851257, -0.51593657,
       -0.50409453, -0.49293061, -0.48239409, -0.47243883, -0.46302284,
       -0.45410781, -0.44565877, -0.43764376, -0.4300335 , -0.42280118,
       -0.41592218, -0.40937391, -0.40313557, -0.39718803, -0.39151365,
       -0.38609616, -0.38092054, -0.3759729 , -0.37124038, -0.36671109,
       -0.362374  , -0.35821887, -0.3542362 , -0.35041717, -0.34675356,
       -0.34323773, -0.33986258, -0.33662146, -0.33350819, -0.33051699,
       -0.32764249, -0.32487965, -0.32222376, -0.31967042, -0.31721551,
       -0.31485518, -0.31258583, -0.31040407, -0.30830674, -0.30629088,
       -0.3043537 , -0.3024926 , -0.30070515, -0.29898905, -0.29734217,
       -0.29576249, -0.29424816, -0.2927974 , -0.29140857, -0.29008015,
       -0.28881071, -0.2875989 , -0.2864435 , -0.28534334, -0.28429737,
       -0.28330459, -0.2823641 , -0.28147506, -0.28063671, -0.27984836,
       -0.27910937, -0.27841919, -0.27777732, -0.27718332, -0.27663681,
       -0.27613748, -0.27568506, -0.27527936, -0.27492022, -0.27460755,
       -0.27434133, -0.27412158, -0.27394837, -0.27382183, -0.27374217,
       -0.27370962, -0.2737245 , -0.27378716, -0.27389805, -0.27405764,
       -0.27426649, -0.27452522, -0.27483451, -0.27519511, -0.27560785,
       -0.27607364, -0.27659345, -0.27716834, -0.27779946, -0.27848804,
       -0.27923542, -0.28004302, -0.28091238, -0.28184513, -0.28284303,
       -0.28390797, -0.28504195, -0.28624713, -0.2875258 , -0.28888041,
       -0.29031359, -0.29182813, -0.29342704, -0.2951135 , -0.29689094,
       -0.29876302, -0.30073363, -0.30280698]

return P0, P1, P2, P3

end
function get_P_tca()
P0 = [-1.        , -0.95133387, -0.90598452, -0.86368159, -0.82418015,
       -0.787258  , -0.75271329, -0.72036246, -0.69003832, -0.66158842,
       -0.63487363, -0.60976676, -0.58615146, -0.56392114, -0.54297806,
       -0.52323253, -0.50460211, -0.48701099, -0.47038937, -0.45467292,
       -0.43980232, -0.42572278, -0.41238369, -0.39973824, -0.38774309,
       -0.37635811, -0.36554607, -0.35527245, -0.34550518, -0.33621445,
       -0.32737256, -0.31895372, -0.31093392, -0.30329076, -0.29600339,
       -0.28905232, -0.28241937, -0.27608755, -0.27004099, -0.26426482,
       -0.25874515, -0.25346895, -0.24842403, -0.24359896, -0.23898303,
       -0.23456616, -0.23033892, -0.22629246, -0.22241844, -0.21870904,
       -0.2151569 , -0.21175513, -0.2084972 , -0.20537701, -0.2023888 ,
       -0.19952716, -0.196787  , -0.19416351, -0.19165219, -0.1892488 ,
       -0.18694933, -0.18475003, -0.18264737, -0.18063803, -0.17871888,
       -0.17688701, -0.17513966, -0.17347426, -0.1718884 , -0.17037983,
       -0.16894645, -0.16758629, -0.16629754, -0.16507851, -0.16392764,
       -0.1628435 , -0.16182477, -0.16087026, -0.15997888, -0.15914966,
       -0.15838173, -0.15767435, -0.15702685, -0.15643871, -0.15590946,
       -0.15543879, -0.15502646, -0.15467235, -0.15437643, -0.15413878,
       -0.15395962, -0.15383924, -0.15377807, -0.15377666, -0.15383565,
       -0.15395585, -0.15413816, -0.15438366, -0.15469353, -0.15506914,
       -0.15551197, -0.15602372, -0.15660624, -0.15726156, -0.15799192,
       -0.15879979, -0.15968783, -0.16065898, -0.16171641, -0.1628636 ,
       -0.16410432, -0.16544266, -0.16688307, -0.16843038, -0.17008985,
       -0.17186716, -0.1737685 , -0.17580059, -0.17797072, -0.18028683,
       -0.18275755, -0.18539225, -0.18820117, -0.19119542, -0.19438715,
       -0.19778961, -0.20141726, -0.20528591]
P1 = [-1.        , -0.95155712, -0.90681708, -0.86542992, -0.82708395,
       -0.79150115, -0.75843305, -0.72765729, -0.69897454, -0.67220581,
       -0.64719014, -0.62378253, -0.60185223, -0.58128106, -0.56196215,
       -0.54379862, -0.52670259, -0.51059421, -0.4954008 , -0.48105616,
       -0.46749987, -0.45467676, -0.44253633, -0.43103234, -0.4201224 ,
       -0.40976756, -0.39993204, -0.39058289, -0.38168977, -0.37322467,
       -0.36516176, -0.35747712, -0.35014864, -0.34315585, -0.33647974,
       -0.33010269, -0.32400832, -0.31818141, -0.31260778, -0.30727425,
       -0.30216849, -0.29727905, -0.2925952 , -0.28810693, -0.28380488,
       -0.27968029, -0.27572497, -0.27193124, -0.26829189, -0.26480019,
       -0.2614498 , -0.25823477, -0.25514952, -0.25218882, -0.24934773,
       -0.24662162, -0.24400615, -0.24149723, -0.239091  , -0.23678384,
       -0.23457237, -0.23245337, -0.23042385, -0.22848098, -0.22662211,
       -0.22484476, -0.22314659, -0.22152541, -0.21997918, -0.21850599,
       -0.21710406, -0.21577171, -0.21450741, -0.21330973, -0.21217734,
       -0.21110903, -0.21010368, -0.20916027, -0.20827787, -0.20745568,
       -0.20669294, -0.20598901, -0.20534334, -0.20475546, -0.20422499,
       -0.20375163, -0.20333518, -0.20297551, -0.20267259, -0.20242647,
       -0.2022373 , -0.20210531, -0.20203083, -0.20201427, -0.20205616,
       -0.2021571 , -0.20231781, -0.20253913, -0.20282198, -0.20316742,
       -0.20357662, -0.20405086, -0.20459159, -0.20520037, -0.20587891,
       -0.20662908, -0.20745291, -0.2083526 , -0.20933056, -0.21038935,
       -0.21153179, -0.21276089, -0.21407991, -0.21549236, -0.21700205,
       -0.21861304, -0.22032976, -0.22215695, -0.22409972, -0.2261636 ,
       -0.22835453, -0.23067894, -0.23314373, -0.23575638, -0.23852497,
       -0.2414582 , -0.24456551, -0.24785709]
P2 = [-1.        , -0.95155712, -0.90681708, -0.86542992, -0.82708395,
       -0.79150115, -0.75843305, -0.72765729, -0.69897454, -0.67220581,
       -0.64719014, -0.62378253, -0.60185223, -0.58128106, -0.56196215,
       -0.54379862, -0.52670259, -0.51059421, -0.4954008 , -0.48105616,
       -0.46749987, -0.45467676, -0.44253633, -0.43103234, -0.4201224 ,
       -0.40976756, -0.39993204, -0.39058289, -0.38168977, -0.37322467,
       -0.36516176, -0.35747712, -0.35014864, -0.34315585, -0.33647974,
       -0.33010269, -0.32400832, -0.31818141, -0.31260778, -0.30727425,
       -0.30216849, -0.29727905, -0.2925952 , -0.28810693, -0.28380488,
       -0.27968029, -0.27572497, -0.27193124, -0.26829189, -0.26480019,
       -0.2614498 , -0.25823477, -0.25514952, -0.25218882, -0.24934773,
       -0.24662162, -0.24400615, -0.24149723, -0.239091  , -0.23678384,
       -0.23457237, -0.23245337, -0.23042385, -0.22848098, -0.22662211,
       -0.22484476, -0.22314659, -0.22152541, -0.21997918, -0.21850599,
       -0.21710406, -0.21577171, -0.21450741, -0.21330973, -0.21217734,
       -0.21110903, -0.21010368, -0.20916027, -0.20827787, -0.20745568,
       -0.20669294, -0.20598901, -0.20534334, -0.20475546, -0.20422499,
       -0.20375163, -0.20333518, -0.20297551, -0.20267259, -0.20242647,
       -0.2022373 , -0.20210531, -0.20203083, -0.20201427, -0.20205616,
       -0.2021571 , -0.20231781, -0.20253913, -0.20282198, -0.20316742,
       -0.20357662, -0.20405086, -0.20459159, -0.20520037, -0.20587891,
       -0.20662908, -0.20745291, -0.2083526 , -0.20933056, -0.21038935,
       -0.21153179, -0.21276089, -0.21407991, -0.21549236, -0.21700205,
       -0.21861304, -0.22032976, -0.22215695, -0.22409972, -0.2261636 ,
       -0.22835453, -0.23067894, -0.23314373, -0.23575638, -0.23852497,
       -0.2414582 , -0.24456551, -0.24785709]
P3 = [-1.        , -0.95178041, -0.9076504 , -0.86718178, -0.82999799,
       -0.79576716, -0.76419626, -0.735026  , -0.7080265 , -0.68299359,
       -0.65974558, -0.63812047, -0.61797357, -0.59917542, -0.58160999,
       -0.5651731 , -0.54977107, -0.53531949, -0.52174221, -0.50897044,
       -0.4969419 , -0.48560013, -0.47489391, -0.46477667, -0.45520601,
       -0.44614328, -0.43755321, -0.42940356, -0.42166483, -0.41430999,
       -0.40731424, -0.4006548 , -0.39431072, -0.38826273, -0.38249306,
       -0.3769853 , -0.37172431, -0.3666961 , -0.36188771, -0.35728713,
       -0.35288324, -0.34866571, -0.34462494, -0.34075202, -0.33703864,
       -0.33347709, -0.33006015, -0.32678111, -0.32363371, -0.3206121 ,
       -0.3177108 , -0.3149247 , -0.31224902, -0.3096793 , -0.30721132,
       -0.30484118, -0.30256519, -0.30037991, -0.29828211, -0.29626875,
       -0.29433699, -0.29248418, -0.2907078 , -0.28900553, -0.28737515,
       -0.28581462, -0.28432201, -0.28289552, -0.28153345, -0.28023423,
       -0.27899639, -0.27781857, -0.27669947, -0.27563793, -0.27463283,
       -0.27368318, -0.27278802, -0.27194651, -0.27115786, -0.27042136,
       -0.26973636, -0.2691023 , -0.26851867, -0.26798503, -0.26750099,
       -0.26706623, -0.26668051, -0.26634361, -0.26605542, -0.26581585,
       -0.26562488, -0.26548256, -0.26538898, -0.26534432, -0.2653488 ,
       -0.26540269, -0.26550635, -0.26566018, -0.26586467, -0.26612035,
       -0.26642784, -0.26678781, -0.26720102, -0.2676683 , -0.26819056,
       -0.26876878, -0.26940404, -0.2700975 , -0.27085041, -0.27166412,
       -0.27254009, -0.27347988, -0.27448516, -0.27555771, -0.27669948,
       -0.27791249, -0.27919896, -0.28056123, -0.28200181, -0.28352337,
       -0.28512878, -0.2868211 , -0.2886036 , -0.29047977, -0.29245335,
       -0.29452833, -0.29670899, -0.2989999 ]

return P0, P1, P2, P3

end