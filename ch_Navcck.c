/* Created by Language version: 7.7.0 */
/* NOT VECTORIZED */
#define NRN_VECTORIZED 0
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ch_Navcck
#define _nrn_initial _nrn_initial__ch_Navcck
#define nrn_cur _nrn_cur__ch_Navcck
#define _nrn_current _nrn_current__ch_Navcck
#define nrn_jacob _nrn_jacob__ch_Navcck
#define nrn_state _nrn_state__ch_Navcck
#define _net_receive _net_receive__ch_Navcck 
#define _f_trates _f_trates__ch_Navcck 
#define rates rates__ch_Navcck 
#define states states__ch_Navcck 
#define trates trates__ch_Navcck 
 
#define _threadargscomma_ /**/
#define _threadargsprotocomma_ /**/
#define _threadargs_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define gmax _p[0]
#define g _p[1]
#define ina _p[2]
#define minf _p[3]
#define hinf _p[4]
#define mtau _p[5]
#define htau _p[6]
#define sexp _p[7]
#define myi _p[8]
#define sinf _p[9]
#define taus _p[10]
#define m _p[11]
#define h _p[12]
#define s _p[13]
#define ena _p[14]
#define Dm _p[15]
#define Dh _p[16]
#define Ds _p[17]
#define mexp _p[18]
#define hexp _p[19]
#define _g _p[20]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_alps(void);
 static void _hoc_alpv(void);
 static void _hoc_bets(void);
 static void _hoc_rates(void);
 static void _hoc_states(void);
 static void _hoc_trates(void);
 static void _hoc_vtrap(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_ch_Navcck", _hoc_setdata,
 "alps_ch_Navcck", _hoc_alps,
 "alpv_ch_Navcck", _hoc_alpv,
 "bets_ch_Navcck", _hoc_bets,
 "rates_ch_Navcck", _hoc_rates,
 "states_ch_Navcck", _hoc_states,
 "trates_ch_Navcck", _hoc_trates,
 "vtrap_ch_Navcck", _hoc_vtrap,
 0, 0
};
#define alps alps_ch_Navcck
#define alpv alpv_ch_Navcck
#define bets bets_ch_Navcck
#define vtrap vtrap_ch_Navcck
 extern double alps( double );
 extern double alpv( double );
 extern double bets( double );
 extern double vtrap( double , double );
 /* declare global and static user variables */
#define ar2 ar2_ch_Navcck
 double ar2 = 0.2;
#define a0s a0s_ch_Navcck
 double a0s = 0.0003;
#define gms gms_ch_Navcck
 double gms = 0.2;
#define hBetaV hBetaV_ch_Navcck
 double hBetaV = 12.5;
#define hBetaC hBetaC_ch_Navcck
 double hBetaC = 1.3;
#define hAlphV hAlphV_ch_Navcck
 double hAlphV = 65;
#define hAlphC hAlphC_ch_Navcck
 double hAlphC = 0.6;
#define mBetaV mBetaV_ch_Navcck
 double mBetaV = 13;
#define mBetaC mBetaC_ch_Navcck
 double mBetaC = 0.3;
#define mAlphV mAlphV_ch_Navcck
 double mAlphV = 42;
#define mAlphC mAlphC_ch_Navcck
 double mAlphC = -0.5;
#define smax smax_ch_Navcck
 double smax = 10;
#define sBetaV sBetaV_ch_Navcck
 double sBetaV = 35;
#define sBetaC sBetaC_ch_Navcck
 double sBetaC = 0.005;
#define sAlphV sAlphV_ch_Navcck
 double sAlphV = 45;
#define sAlphC sAlphC_ch_Navcck
 double sAlphC = 0.003;
#define sh sh_ch_Navcck
 double sh = 15;
#define usetable usetable_ch_Navcck
 double usetable = 1;
#define vvs vvs_ch_Navcck
 double vvs = 2;
#define vvh vvh_ch_Navcck
 double vvh = -58;
#define vhalfs vhalfs_ch_Navcck
 double vhalfs = -60;
#define zetas zetas_ch_Navcck
 double zetas = 12;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_ch_Navcck", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "sh_ch_Navcck", "mV",
 "mAlphC_ch_Navcck", "1",
 "mAlphV_ch_Navcck", "mV",
 "mBetaC_ch_Navcck", "1",
 "mBetaV_ch_Navcck", "mV",
 "hAlphC_ch_Navcck", "1",
 "hAlphV_ch_Navcck", "mV",
 "hBetaC_ch_Navcck", "1",
 "hBetaV_ch_Navcck", "mV",
 "sAlphC_ch_Navcck", "1",
 "sAlphV_ch_Navcck", "mV",
 "sBetaC_ch_Navcck", "1",
 "sBetaV_ch_Navcck", "mV",
 "vhalfs_ch_Navcck", "mV",
 "a0s_ch_Navcck", "ms",
 "zetas_ch_Navcck", "1",
 "gms_ch_Navcck", "1",
 "smax_ch_Navcck", "ms",
 "vvh_ch_Navcck", "mV",
 "vvs_ch_Navcck", "mV",
 "ar2_ch_Navcck", "1",
 "gmax_ch_Navcck", "mho/cm2",
 "g_ch_Navcck", "mho/cm2",
 "ina_ch_Navcck", "mA/cm2",
 "mtau_ch_Navcck", "ms",
 "htau_ch_Navcck", "ms",
 "myi_ch_Navcck", "mA/cm2",
 "taus_ch_Navcck", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double s0 = 0;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "sh_ch_Navcck", &sh_ch_Navcck,
 "mAlphC_ch_Navcck", &mAlphC_ch_Navcck,
 "mAlphV_ch_Navcck", &mAlphV_ch_Navcck,
 "mBetaC_ch_Navcck", &mBetaC_ch_Navcck,
 "mBetaV_ch_Navcck", &mBetaV_ch_Navcck,
 "hAlphC_ch_Navcck", &hAlphC_ch_Navcck,
 "hAlphV_ch_Navcck", &hAlphV_ch_Navcck,
 "hBetaC_ch_Navcck", &hBetaC_ch_Navcck,
 "hBetaV_ch_Navcck", &hBetaV_ch_Navcck,
 "sAlphC_ch_Navcck", &sAlphC_ch_Navcck,
 "sAlphV_ch_Navcck", &sAlphV_ch_Navcck,
 "sBetaC_ch_Navcck", &sBetaC_ch_Navcck,
 "sBetaV_ch_Navcck", &sBetaV_ch_Navcck,
 "vhalfs_ch_Navcck", &vhalfs_ch_Navcck,
 "a0s_ch_Navcck", &a0s_ch_Navcck,
 "zetas_ch_Navcck", &zetas_ch_Navcck,
 "gms_ch_Navcck", &gms_ch_Navcck,
 "smax_ch_Navcck", &smax_ch_Navcck,
 "vvh_ch_Navcck", &vvh_ch_Navcck,
 "vvs_ch_Navcck", &vvs_ch_Navcck,
 "ar2_ch_Navcck", &ar2_ch_Navcck,
 "usetable_ch_Navcck", &usetable_ch_Navcck,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ch_Navcck",
 "gmax_ch_Navcck",
 0,
 "g_ch_Navcck",
 "ina_ch_Navcck",
 "minf_ch_Navcck",
 "hinf_ch_Navcck",
 "mtau_ch_Navcck",
 "htau_ch_Navcck",
 "sexp_ch_Navcck",
 "myi_ch_Navcck",
 "sinf_ch_Navcck",
 "taus_ch_Navcck",
 0,
 "m_ch_Navcck",
 "h_ch_Navcck",
 "s_ch_Navcck",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 21, _prop);
 	/*initialize range parameters*/
 	gmax = 0;
 	_prop->param = _p;
 	_prop->param_size = 21;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ch_Navcck_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("na", 1.0);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 21, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ch_Navcck C:/Users/ddopp/source/repos/CA1_model/biophys_components/mechanisms/modfiles/ch_Navcck.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96520.0;
 static double R = 8.3134;
 static double _zq10 ;
 static double *_t_minf;
 static double *_t_mexp;
 static double *_t_hinf;
 static double *_t_hexp;
 static double *_t_mtau;
 static double *_t_htau;
 static double *_t_sinf;
 static double *_t_taus;
 static double *_t_sexp;
static int _reset;
static char *modelname = "sodium channel (voltage dependent)";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_trates(double);
static int rates(double);
static int states();
static int trates(double);
 static void _n_trates(double);
 
/*VERBATIM*/
#include <stdlib.h> 
/* 	Include this library so that the following (innocuous) warning does not appear:
		In function '_thread_cleanup':
		warning: incompatible implicit declaration of built-in function 'free'  */
 
static int  states (  ) {
   trates ( _threadargscomma_ v ) ;
   m = m + mexp * ( minf - m ) ;
   h = h + hexp * ( hinf - h ) ;
   s = s + sexp * ( sinf - s ) ;
    return 0; }
 
static void _hoc_states(void) {
  double _r;
   _r = 1.;
 states (  );
 hoc_retpushx(_r);
}
 
double alpv (  double _lv ) {
   double _lalpv;
 _lalpv = 1.0 / ( 1.0 + exp ( ( _lv - vvh - sh ) / vvs ) ) ;
   
return _lalpv;
 }
 
static void _hoc_alpv(void) {
  double _r;
   _r =  alpv (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double alps (  double _lv ) {
   double _lalps;
 _lalps = exp ( 1.e-3 * zetas * ( _lv - vhalfs - sh ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lalps;
 }
 
static void _hoc_alps(void) {
  double _r;
   _r =  alps (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double bets (  double _lv ) {
   double _lbets;
 _lbets = .2 * exp ( 1.e-3 * zetas * gms * ( _lv - vhalfs - sh ) * 9.648e4 / ( 8.315 * ( 273.16 + celsius ) ) ) ;
   
return _lbets;
 }
 
static void _hoc_bets(void) {
  double _r;
   _r =  bets (  *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  rates (  double _lv ) {
   double _lalpha , _lbeta , _lc , _lsum ;
 _zq10 = pow( 3.0 , ( ( celsius - 34.0 ) / 10.0 ) ) ;
   _lalpha = mAlphC * vtrap ( _threadargscomma_ ( _lv + mAlphV ) , - 5.0 ) ;
   _lbeta = mBetaC * vtrap ( _threadargscomma_ ( _lv + mBetaV ) , 5.0 ) ;
   _lsum = _lalpha + _lbeta ;
   mtau = 1.0 / _lsum ;
   minf = _lalpha / _lsum ;
   _lalpha = hAlphC / exp ( ( _lv + hAlphV ) / 20.0 ) ;
   _lbeta = hBetaC / ( 1.0 + exp ( ( _lv + hBetaV ) / - 10.0 ) ) ;
   _lsum = _lalpha + _lbeta ;
   htau = 1.0 / _lsum ;
   hinf = _lalpha / _lsum ;
   _lalpha = sAlphC / exp ( ( _lv + sAlphV ) / 6.0 ) ;
   _lbeta = sBetaC / ( 1.0 + exp ( ( _lv + sBetaV ) / - 20.0 ) ) ;
   _lsum = _lalpha + _lbeta ;
   taus = 1.0 / _lsum ;
   sinf = _lalpha / _lsum ;
   if ( taus < smax ) {
     taus = smax ;
     }
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 static double _mfac_trates, _tmin_trates;
 static void _check_trates();
 static void _check_trates() {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_dt;
  static double _sav_celsius;
  static double _sav_mAlphV;
  static double _sav_mAlphC;
  static double _sav_mBetaV;
  static double _sav_mBetaC;
  static double _sav_hAlphV;
  static double _sav_hAlphC;
  static double _sav_hBetaV;
  static double _sav_hBetaC;
  static double _sav_sAlphV;
  static double _sav_sAlphC;
  static double _sav_sBetaV;
  static double _sav_sBetaC;
  if (!usetable) {return;}
  if (_sav_dt != dt) { _maktable = 1;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_sav_mAlphV != mAlphV) { _maktable = 1;}
  if (_sav_mAlphC != mAlphC) { _maktable = 1;}
  if (_sav_mBetaV != mBetaV) { _maktable = 1;}
  if (_sav_mBetaC != mBetaC) { _maktable = 1;}
  if (_sav_hAlphV != hAlphV) { _maktable = 1;}
  if (_sav_hAlphC != hAlphC) { _maktable = 1;}
  if (_sav_hBetaV != hBetaV) { _maktable = 1;}
  if (_sav_hBetaC != hBetaC) { _maktable = 1;}
  if (_sav_sAlphV != sAlphV) { _maktable = 1;}
  if (_sav_sAlphC != sAlphC) { _maktable = 1;}
  if (_sav_sBetaV != sBetaV) { _maktable = 1;}
  if (_sav_sBetaC != sBetaC) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_trates =  - 100.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_trates)/200.; _mfac_trates = 1./_dx;
   for (_i=0, _x=_tmin_trates; _i < 201; _x += _dx, _i++) {
    _f_trates(_x);
    _t_minf[_i] = minf;
    _t_mexp[_i] = mexp;
    _t_hinf[_i] = hinf;
    _t_hexp[_i] = hexp;
    _t_mtau[_i] = mtau;
    _t_htau[_i] = htau;
    _t_sinf[_i] = sinf;
    _t_taus[_i] = taus;
    _t_sexp[_i] = sexp;
   }
   _sav_dt = dt;
   _sav_celsius = celsius;
   _sav_mAlphV = mAlphV;
   _sav_mAlphC = mAlphC;
   _sav_mBetaV = mBetaV;
   _sav_mBetaC = mBetaC;
   _sav_hAlphV = hAlphV;
   _sav_hAlphC = hAlphC;
   _sav_hBetaV = hBetaV;
   _sav_hBetaC = hBetaC;
   _sav_sAlphV = sAlphV;
   _sav_sAlphC = sAlphC;
   _sav_sBetaV = sBetaV;
   _sav_sBetaC = sBetaC;
  }
 }

 static int trates(double _lv){ _check_trates();
 _n_trates(_lv);
 return 0;
 }

 static void _n_trates(double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_trates(_lv); return; 
}
 _xi = _mfac_trates * (_lv - _tmin_trates);
 if (isnan(_xi)) {
  minf = _xi;
  mexp = _xi;
  hinf = _xi;
  hexp = _xi;
  mtau = _xi;
  htau = _xi;
  sinf = _xi;
  taus = _xi;
  sexp = _xi;
  return;
 }
 if (_xi <= 0.) {
 minf = _t_minf[0];
 mexp = _t_mexp[0];
 hinf = _t_hinf[0];
 hexp = _t_hexp[0];
 mtau = _t_mtau[0];
 htau = _t_htau[0];
 sinf = _t_sinf[0];
 taus = _t_taus[0];
 sexp = _t_sexp[0];
 return; }
 if (_xi >= 200.) {
 minf = _t_minf[200];
 mexp = _t_mexp[200];
 hinf = _t_hinf[200];
 hexp = _t_hexp[200];
 mtau = _t_mtau[200];
 htau = _t_htau[200];
 sinf = _t_sinf[200];
 taus = _t_taus[200];
 sexp = _t_sexp[200];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 minf = _t_minf[_i] + _theta*(_t_minf[_i+1] - _t_minf[_i]);
 mexp = _t_mexp[_i] + _theta*(_t_mexp[_i+1] - _t_mexp[_i]);
 hinf = _t_hinf[_i] + _theta*(_t_hinf[_i+1] - _t_hinf[_i]);
 hexp = _t_hexp[_i] + _theta*(_t_hexp[_i+1] - _t_hexp[_i]);
 mtau = _t_mtau[_i] + _theta*(_t_mtau[_i+1] - _t_mtau[_i]);
 htau = _t_htau[_i] + _theta*(_t_htau[_i+1] - _t_htau[_i]);
 sinf = _t_sinf[_i] + _theta*(_t_sinf[_i+1] - _t_sinf[_i]);
 taus = _t_taus[_i] + _theta*(_t_taus[_i+1] - _t_taus[_i]);
 sexp = _t_sexp[_i] + _theta*(_t_sexp[_i+1] - _t_sexp[_i]);
 }

 
static int  _f_trates (  double _lv ) {
   double _ltinc ;
 rates ( _threadargscomma_ _lv ) ;
   _ltinc = - dt * _zq10 ;
   mexp = 1.0 - exp ( _ltinc / mtau ) ;
   hexp = 1.0 - exp ( _ltinc / htau ) ;
   sexp = 1.0 - exp ( _ltinc / taus ) ;
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
    _r = 1.;
 trates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
double vtrap (  double _lx , double _ly ) {
   double _lvtrap;
 if ( fabs ( _lx / _ly ) < 1e-6 ) {
     _lvtrap = _ly * ( 1.0 - _lx / _ly / 2.0 ) ;
     }
   else {
     _lvtrap = _lx / ( exp ( _lx / _ly ) - 1.0 ) ;
     }
   
return _lvtrap;
 }
 
static void _hoc_vtrap(void) {
  double _r;
   _r =  vtrap (  *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("ch_Navcck", "cannot be used with CVODE"); return 0;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  h = h0;
  m = m0;
  s = s0;
 {
   trates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   s = sinf ;
   }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gmax * m * m * m * h * s ;
   ina = g * ( v - ena ) ;
   myi = ina ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
 _g = _nrn_current(_v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  ena = _ion_ena;
 { error =  states();
 if(error){fprintf(stderr,"at line 98 in file ch_Navcck.mod:\n	SOLVE states\n"); nrn_complain(_p); abort_run(error);}
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
   _t_minf = makevector(201*sizeof(double));
   _t_mexp = makevector(201*sizeof(double));
   _t_hinf = makevector(201*sizeof(double));
   _t_hexp = makevector(201*sizeof(double));
   _t_mtau = makevector(201*sizeof(double));
   _t_htau = makevector(201*sizeof(double));
   _t_sinf = makevector(201*sizeof(double));
   _t_taus = makevector(201*sizeof(double));
   _t_sexp = makevector(201*sizeof(double));
_first = 0;
}

#if NMODL_TEXT
static const char* nmodl_filename = "ch_Navcck.mod";
static const char* nmodl_file_text = 
  "TITLE sodium channel (voltage dependent)\n"
  "\n"
  "COMMENT\n"
  "sodium channel (voltage dependent)\n"
  "\n"
  "Ions: na\n"
  "\n"
  "Style: quasi-ohmic\n"
  "\n"
  "From: not sure where from\n"
  "\n"
  "Updates:\n"
  "2014 December (Marianne Bezaire): documented\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "VERBATIM\n"
  "#include <stdlib.h> \n"
  "/* 	Include this library so that the following (innocuous) warning does not appear:\n"
  "		In function '_thread_cleanup':\n"
  "		warning: incompatible implicit declaration of built-in function 'free'  */\n"
  "ENDVERBATIM\n"
  "\n"
  "UNITS {\n"
  "	(mA) =(milliamp)\n"
  "	(mV) =(millivolt)\n"
  "	(uF) = (microfarad)\n"
  "	(molar) = (1/liter)\n"
  "	(nA) = (nanoamp)\n"
  "	(mM) = (millimolar)\n"
  "	(um) = (micron)\n"
  "	FARADAY = 96520 (coul)\n"
  "	R = 8.3134	(joule/degC)\n"
  "}\n"
  " \n"
  "NEURON { \n"
  "	SUFFIX ch_Navcck\n"
  "	USEION na READ ena WRITE ina VALENCE 1\n"
  "	RANGE g, gmax, minf, mtau, hinf, htau, ina, m, h, sinf, taus, s, sexp\n"
  "	RANGE myi\n"
  "	THREADSAFE\n"
  "}\n"
  " \n"
  "PARAMETER {\n"
  "	ena  (mV)\n"
  "	gmax (mho/cm2)   \n"
  "	sh   = 15		(mV)\n"
  "\n"
  "	mAlphC = -0.5 (1) : -0.6 (1)\n"
  "	mAlphV = 42 (mV) : 44 (mV)\n"
  "	mBetaC = 0.3 (1) : 0.3\n"
  "	mBetaV = 13.0 (mV) : 14.5 (mV)\n"
  "\n"
  "	hAlphC = 0.6 (1) : 23\n"
  "	hAlphV = 65 (mV)\n"
  "	hBetaC = 1.3 (1) : 3.33\n"
  "	hBetaV = 12.5 (mV)\n"
  "\n"
  "	sAlphC = 0.003 (1) : 0.04 (1) : 23\n"
  "	sAlphV = 45 (mV) : 45 (mV)\n"
  "	sBetaC = 0.005 (1) :1 (1) : 3.33\n"
  "	sBetaV = 35 (mV) : 12.5 (mV)\n"
  "\n"
  "	vhalfs = -60	(mV)	: slow inact.\n"
  "	a0s = 0.0003	(ms)	: a0s=b0s\n"
  "	zetas = 12		(1)\n"
  "	gms = 0.2		(1)\n"
  "	smax = 10		(ms)\n"
  "	vvh = -58		(mV) \n"
  "	vvs = 2			(mV)\n"
  "	ar2 = 0.2			(1)		: 1=no inact., 0=max inact.\n"
  "}\n"
  " \n"
  "STATE {\n"
  "	m h s\n"
  "}\n"
  " \n"
  "ASSIGNED {\n"
  "	v (mV) \n"
  "	celsius (degC) : temperature - set in hoc; default is 6.3\n"
  "	dt (ms) \n"
  "\n"
  "	g (mho/cm2)\n"
  "	ina (mA/cm2)\n"
  "	minf\n"
  "	hinf\n"
  "	mtau (ms)\n"
  "	htau (ms)\n"
  "	mexp\n"
  "	hexp \n"
  "	sexp \n"
  "	myi (mA/cm2)\n"
  "	sinf 	\n"
  "	taus		(ms)\n"
  "} \n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states\n"
  "	g = gmax*m*m*m*h*s :(1-(1-s)*.5)\n"
  "	ina = g*(v - ena)\n"
  "	myi = ina\n"
  "}\n"
  " \n"
  "UNITSOFF\n"
  " \n"
  "INITIAL {\n"
  "	trates(v)\n"
  "	m = minf\n"
  "	h = hinf\n"
  "	s = sinf\n"
  "}\n"
  "\n"
  "PROCEDURE states() {	:Computes state variables m, h, and n \n"
  "	trates(v)			:      at the current v and dt.\n"
  "	m = m + mexp*(minf-m)\n"
  "	h = h + hexp*(hinf-h)\n"
  "	:printf(\"t=%.2f, s=%.3f, sinf=%.3f, sexp=%.4f\\n\", t, s, sinf, sexp)\n"
  "	s = s + sexp*(sinf-s)\n"
  "}\n"
  "\n"
  "FUNCTION alpv(v(mV)) {\n"
  "	alpv = 1/(1+exp((v-vvh-sh)/vvs))\n"
  "}\n"
  "        \n"
  "FUNCTION alps(v(mV)) {  \n"
  "	alps = exp(1.e-3*zetas*(v-vhalfs-sh)*9.648e4/(8.315*(273.16+celsius)))\n"
  "}\n"
  "\n"
  "FUNCTION bets(v(mV)) {\n"
  "	bets = .2*exp(1.e-3*zetas*gms*(v-vhalfs-sh)*9.648e4/(8.315*(273.16+celsius)))\n"
  "}\n"
  " \n"
  "LOCAL q10	: declare outside a block so available to whole mechanism\n"
  "PROCEDURE rates(v) {  :Computes rate and other constants at current v.\n"
  "                      :Call once from HOC to initialize inf at resting v.\n"
  "	LOCAL  alpha, beta, c, sum	: only available to block; must be first line in block\n"
  "\n"
  "	q10 = 3^((celsius - 34)/10)\n"
  "\n"
  "	:\"m\" sodium activation system - act and inact cross at -40\n"
  "	alpha = mAlphC*vtrap((v+mAlphV),-5)\n"
  "	beta = mBetaC*vtrap((v+mBetaV),5)\n"
  "	sum = alpha+beta        \n"
  "	mtau = 1/sum \n"
  "	minf = alpha/sum\n"
  "	\n"
  "	:\"h\" sodium inactivation system\n"
  "	alpha = hAlphC/exp((v+hAlphV)/20)\n"
  "	beta = hBetaC/(1+exp((v+hBetaV)/-10))\n"
  "	sum = alpha+beta\n"
  "	htau = 1/sum \n"
  "	hinf = alpha/sum 		\n"
  "	\n"
  "	:\"s\" slow sodium inactivation system\n"
  "	alpha = sAlphC/exp((v+sAlphV)/6)  : 10\n"
  "	beta = sBetaC/(1+exp((v+sBetaV)/-20)) : -10\n"
  "	sum = alpha+beta\n"
  "	taus = 1/sum \n"
  "	sinf = alpha/sum \n"
  "\n"
  "	:c=alpv(v)\n"
  "	:sinf = c+ar2*(1-c)\n"
  "	:taus = bets(v)/(a0s*(1+alps(v)))\n"
  "	if (taus<smax) {taus=smax}\n"
  "}\n"
  " \n"
  "PROCEDURE trates(v) {  :Computes rate and other constants at current v.\n"
  "                      :Call once from HOC to initialize inf at resting v.\n"
  "	LOCAL tinc	: only available to block; must be first line in block\n"
  "	TABLE minf, mexp, hinf, hexp, mtau, htau, sinf, taus, sexp\n"
  "	DEPEND dt, celsius, mAlphV, mAlphC, mBetaV, mBetaC, hAlphV, hAlphC, hBetaV, hBetaC, sAlphV, sAlphC, sBetaV, sBetaC\n"
  "	FROM -100 TO 100 WITH 200\n"
  "                                   \n"
  "	rates(v)	: not consistently executed from here if usetable_hh == 1\n"
  "				: so don't expect the tau values to be tracking along with\n"
  "				: the inf values in hoc\n"
  "\n"
  "	tinc = -dt * q10\n"
  "\n"
  "	mexp = 1 - exp(tinc/mtau)\n"
  "	hexp = 1 - exp(tinc/htau)\n"
  "	sexp = 1 - exp(tinc/taus)\n"
  "	:printf(\"t=%.2f, taus=%.2f, sinf=%.2f\\n\", t, taus, sinf)\n"
  "	:if (t>100.0 && t<100.3) {printf(\"t=%.2f, taus=%.2f, sinf=%.2f\\n\", t, taus, sinf)}\n"
  "	:if (t>900.0 && t<900.3) {printf(\"t=%.2f, taus=%.2f, sinf=%.2f\\n\", t, taus, sinf)}\n"
  "}\n"
  " \n"
  "FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.\n"
  "	if (fabs(x/y) < 1e-6) {\n"
  "		vtrap = y*(1 - x/y/2)\n"
  "	}else{  \n"
  "		vtrap = x/(exp(x/y) - 1)\n"
  "	}\n"
  "}\n"
  " \n"
  "UNITSON\n"
  "\n"
  ;
#endif
