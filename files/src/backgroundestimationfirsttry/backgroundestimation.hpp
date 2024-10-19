#ifndef BACKGROUNDESTIMATION_HPP
#define BACKGROUNDESTIMATION_HPP

#include <cmath>
#include "labConverter.hpp"



lab averageLAB(lab p1, lab p2);
float labDistance(lab p1, lab p2);
void check_background(ImageView<lab> in);

#endif
