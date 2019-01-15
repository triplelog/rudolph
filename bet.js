return function f(ars,ara,hrs,hra) {
  var probA = (Math.pow(ars+10,1.8))/(Math.pow(ars+10,1.8)+Math.pow(ara+10,1.8));
  var probH = (Math.pow(hrs+10,1.8))/(Math.pow(hrs+10,1.8)+Math.pow(hra+10,1.8));
  return (probH-probA*probH)/(probA+probH-2*probA*probH);
};
