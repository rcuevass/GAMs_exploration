# Simple example of GAM

install.packages("faraway")
require(faraway)
data(ozone)

require(mgcv)

head(ozone,10)
plot(log(ozone$O3))

log(ozone$O3)

gamobj<-gam(log(O3)~s(vh)+s(wind)+s(humidity)+s(temp)+s(ibh)+
              s(dpg)+s(ibt)+s(vis)+s(doy),
            family=gaussian(link=identity),data=ozone)

summary(gamobj)

ls(gamobj)
gamobj$formula
gamobj$coefficients
gamobj$pred.formula
gamobj$boundary


plot(gamobj)
