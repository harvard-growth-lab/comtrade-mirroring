clear all
set more off
set scheme s1color
set matsize 10000
mata mata clear

*-----------------------------------------------------------------------------------------------
* Set global paths and directories
//global path "D:/Data/Trade_data"
//global path "/Users/sbustos/Dropbox/datasets/Trade_data" 
//global path "/nfs/projects_nobackup/c/cid_trade/Trade_data"
global path "/n/hausmann_lab/lab/atlas/bustos_yildirim/stata_files"
cd "${path}/"

*-----------------------------------------------------------------------------------------------
* 
//loc start = 1962 // 1962
//loc end 2018 // 2015
loc niter 25 // Interations A_e
loc rnflows 10 // 30
loc limit  10^4 // minimum value to assume that there is a flow
loc vfile 1 
loc poplimit 0.5*10^6 //  only include countries with population above this number 
loc anorm 0 // normalize the score
loc alog 0 //  apply logs 
loc af 0  // combine A_e and A_i in single measure
loc seed 1 //  initial value for the A's
*----------------------------------------------------------

 
//=========================================================================
/*
cap use iso year sp_pop_totl  using "/Users/sbustos/Dropbox/datasets/WDI/wdi_extended.dta", clear
cap use iso year sp_pop_totl  using "${path}/Support_Files/wdi_extended.dta", clear
drop if   sp_pop_totl==.
format sp_pop_totl %15.0fc 
rename sp_pop_totl pop_exporter 
rename iso exporter 
save temp_popexporter.dta, replace 
rename exporter importer
rename pop_exporter pop_importer
save temp_popimporter.dta, replace 
*/
//cap use iso year fp_cpi_totl_zg using "/Users/sbustos/Dropbox/datasets/WDI/wdi_extended.dta", clear
cap use iso year fp_cpi_totl_zg using "/n/hausmann_lab/lab/atlas/bustos_yildirim/stata_files"
cap use iso year fp_cpi_totl_zg using "${path}/Support_Files/wdi_extended.dta", clear
rename fp_cpi_totl_zg dcpi
keep if year>=1962
keep if iso=="USA"
tsset year
gen index = 100 if year==1962
qui sum year
loc end = r(max)
 
* inflation adjustment
forval y=1963/`end' {

dcpi = df[df.year == y]['dcpi']
	qui replace index = (l1.index)*(1+dcpi/100) if year==`y'
}
keep if year>=1962 & year<=`end'
sum index if year==2010
replace index  = 1* (index/r(mean))
keep year index
tsset year

replace index = l1.index if index==.
//drop index
//gen index==100
//replace index = 100 if index==.
list year index
save temp_index.dta, replace
//=========================================================================

use year using "Totals_trade.dta", clear
qui sum year
loc start = r(min)
loc end = r(max)

clear
/*
loc start 1990
loc end 1991
*/

*-------------------------------------------------------------------------------	
* 	
forval y = `start'/`end' {
				quietly {
						cd "$path/"
						use "${path}/Totals_trade.dta", clear
						noi di "Doing year = " `y'
						keep if year == `y'
						drop if exporter=="WLD" | importer=="WLD"
						drop if exporter=="nan" | importer=="nan" 
						drop if exporter=="ANS" & importer=="ANS" 
						
						drop if exporter==importer 
						
						* drops max values under 10K
						* takes greater of import or export value
						egen temp = rowmax(importvalue_fob exportvalue_fob)
						drop if temp<10^4
						* cap is error handling if temp doesn't exist
						cap drop temp
						
						preserve 
							* gen (new variable) relative markup between cif (cost, insurance, freight) and fob (value of good at export)
							
							gen temp  = importvalue_cif / importvalue_fob -1
							* group by exporter and importer
							* ?? ex All CHL / USA calculates mean of CIF markup
							egen cif_ratio = mean(temp), by(exporter importer)
							* max reasonable markup value is 20%
							replace cif_ratio = 0.20 if cif_ratio>0.20
							keep if year==`y'
							
							keep year exporter importer   cif_ratio importvalue_fob exportvalue_fob 
							save "temp_accuracy.dta", replace 
						* back to original dataset
						restore 
						
						* ?? how does fillin operation work?
						fillin year exporter importer
						drop _fillin
						/* stop 1
						foreach i in importvalue_fob exportvalue_fob {
								replace `i' = . if `i'<10^3
						}
						*/

						mata mata clear
									
									keep  year exporter importer exportvalue_fob importvalue_cif importvalue_fob
									replace exporter="ROM" if exporter=="ROU" 
									replace importer="ROM" if importer=="ROU" 
									* ?? many to one merge 
									foreach i in exporter importer {
										merge m:1 year `i' using temp_pop`i'
										drop if _merge == 2
										drop _merge 
									}
									replace exporter="ROU" if exporter=="ROM" 
									replace importer="ROU" if importer=="ROM" 

									//--------------------------------------------------------
									// countries without population in WDI, but surely large enough to be included 
									foreach n in YUG DDR SUN CSK ANS TWN {
											replace pop_exporter = 10^7 if exporter=="`n'"
											replace pop_importer = 10^7 if importer=="`n'"
									}
									foreach n in MAC nan NAN ANS { // this drops Macao which appears in the data from 2008-2015
											drop if exporter=="`n'" | importer=="`n'" 
									}
									
									//-------------------------------------------------------
									keep if pop_exporter>`poplimit'
									keep if pop_importer>`poplimit'
									drop  pop_exporter pop_importer
									
									
									fillin year exporter importer
									drop _fillin
									
									cap drop _merge 
									merge m:1 year using temp_index.dta, nogen keep(3)
									
									foreach j in exportvalue_fob importvalue_fob importvalue_cif {
										replace `j' = `j' / (index) 
										replace `j' = 0 if `j' == .
									}		
									
									drop index
									rename exportvalue_fob v_e
									rename importvalue_fob v_i
									//recast float v_e v_i, force

							foreach j in e i {
								replace v_`j' = 0 if v_`j' < `limit'
							}
							
							egen temp1 = total((v_e>0)), by(exporter)
							drop if temp1==0
							cap drop temp*

							egen temp1 = total((v_i>0)), by(importer)
							drop if temp1==0
							cap drop temp*
							sort exporter importer
							
							keep  exporter importer v_*

							*--------------------------------------------------------
							// deviation scores 
							gen s_ij = (abs(v_e - v_i)) / ( (v_e + v_i))
							replace s_ij = 0 if s_ij ==.
							
							// eliminate exporters below certain threshold of flows
							foreach direction in exporter importer {				
								forval t = 1/5 {
									egen nflows = sum( (v_e!=0 | v_i!=0)), by(`direction')
									qui levelsof `direction' if  nflows<`rnflows', local(listctry)
									foreach i of local listctry {
										qui drop if exporter=="`i'" | importer=="`i'" 
									}
									*--------------------------------------------------------
									// make sure that importers and exporters are the same countries
									gen temp=0
									qui levelsof `direction', local(listctry)
									if "`direction'" == "exporter" {  
										foreach i of local listctry {
											qui replace temp = 1 if importer=="`i'"  // importer exporter
										}
									}				
									if "`direction'" == "importer" {  
										foreach i of local listctry {
											qui replace temp = 1 if exporter=="`i'"  // importer exporter
										}
									}
									drop if temp==0
									drop temp
									drop nflows 
								}	
							}
							
							cap drop temp*
							gen temp1 = v_e if v_e!=0
							gen temp2 = v_i if v_i!=0
							egen temp3 = rowmean(temp*)
							egen temp4 = total(temp3), by(exporter)
							egen temp5 = total(temp3), by(importer)
							gen p_e = max(temp3/temp4,0)
							gen p_i = max(temp3/temp5,0)
							cap drop temp*
							
							*--------------------------------------------------------
							foreach direction in exporter importer {	
								egen nflows_`direction' = sum( (v_e!=0 | v_i!=0)), by(`direction')
								egen s_ij_`direction' = mean(s_ij), by(`direction')  
								egen avs_ij_`direction' = total(s_ij_`direction'), by(`direction')
								replace avs_ij_`direction' = avs_ij_`direction'/nflows_`direction'
							}	
							*--------------------------------------------------------
							egen exp = group(exporter)
							egen imp = group(importer)
							*--------------------------------------------------------

							// Exporter //----------------------
							
							preserve 
									sort exp imp
									qui sum exp 
									local N=r(max)
                                    noi di "THIS IS N!!!! IN PRESERVED EXP IMP"
									noi di  `N'
									putmata s_ij nflows_exporter p_e, replace
									mata mata rename s_ij es_ij 
									mata mata rename nflows_exporter en_ij
							restore 
							*------------------------------------------------------------------------------- 
							// Importer //----------------------
							preserve 
									sort imp exp
									qui sum imp 
									local N=r(max)
									putmata s_ij nflows_importer p_i, replace
									mata mata rename s_ij is_ij 
									mata mata rename nflows_importer in_ij
							restore 
							
							
							foreach j in e i {
								foreach i in `j's_ij  `j'n_ij  {
									mata  `i' = colshape(`i', `N')
								}
								mata  p_`j' = colshape(p_`j', `N')
							}
							
							foreach j in en_ij in_ij {
								mata `j' = `j'[.,1]
							}
							
							****************************************************************************
							////// MAY NEW CODE
							
							preserve
								drop if v_e == 0 & v_i == 0
								drop if exporter == importer
								sort exp imp
								putmata exp imp s_ij, replace
							
								egen t_e = tag(exporter)
								keep if t_e
								cap sum exp
								local n_c = r(max)
								mata n_c = `n_c'
								local potential_skipped
								foreach cntry in "USA" "DEU" "GBR" "FRA" "ITA" "JPN" "NLD" "BEL" "CAN" {
									sum exp if exporter == "`cntry'"
									local skipped = r(min)
									if(`skipped' != .) {
										local potential_skipped `potential_skipped' `skipped'
									}
								}
							restore
							
							mata trans_all = J( rows(s_ij) , n_c, 0)
							mata r2_pos = r2_all = J(n_c, 1, 0)
									
							forvalues j = 1/`n_c' {
								mata trans_all[.,`j'] = ((exp :== `j') + (imp :== `j')) :> 0
							}
							
							*forvalues skipped = 1/`n_c' {		
							foreach skipped in `potential_skipped' {
								mata skipped = `skipped'
								mata trans = trans_all
								mata trans[.,skipped] = J( rows(s_ij) , 1, 0)

								mata t_fixed = select(trans, colsum(trans) :> 0)
								mata ols =  (luinv(t_fixed'*t_fixed)*(t_fixed'*s_ij))
								
								mata yhat = t_fixed * ols
								mata r2_all[skipped] = 1 - ((s_ij - yhat)'*(s_ij - yhat)) / ((s_ij :- mean(s_ij))'*(s_ij :- mean(s_ij)))
								
								mata yhat = t_fixed * (ols :* (ols :> 0))
								mata r2_pos[skipped] = 1 - ((s_ij - yhat)'*(s_ij - yhat)) / ((s_ij :- mean(s_ij))'*(s_ij :- mean(s_ij)))
							}
							
							mata skipped = temp = .
							mata maxindex(r2_pos, 1, skipped, temp)
							
							mata trans = trans_all
							mata trans[.,skipped] = J( rows(s_ij) , 1, 0)
							mata select_all = (colsum(trans) :> 0)
							mata t_fixed = select(trans, select_all)
							mata selected_part_index = select((1..n_c),select_all)
							
							mata ols =  (luinv(t_fixed'*t_fixed)*(t_fixed'*s_ij))
							mata sigmas = sigma_se = J(n_c, 1, 0)
							mata sigmas[selected_part_index] = ols
							mata sigma_se[selected_part_index] = diagonal(((s_ij'*s_ij-ols'*t_fixed'*s_ij)/(rows(s_ij)-cols(t_fixed)))*luinv(t_fixed'*t_fixed)):^0.5		
							
							//// MAY NEW CODE FINISHED
							****************************************************************************
							
							// es_ij  en_ij is_ij in_ij
							mata A_e = J(`N',1,`seed')
							mata A_i = A_e
							
							forval i=1/`niter' {
								mata prA_e  =  1 :/ ( ( es_ij * A_i ) :/ en_ij)
								mata prA_i  =  1 :/ ( ( is_ij * A_e ) :/ in_ij)
								mata A_e = prA_e
								mata A_i = prA_i
							}	

							mata es_ij = rowsum(es_ij) :/ `N'
							mata is_ij = rowsum(is_ij) :/ `N'
							
							
							egen tag = tag(exporter)
							keep if tag
							rename exporter iso
							gen year = `y'
							keep year iso
							
							
							getmata A_e A_i en_ij in_ij es_ij is_ij sigmas 
							 
							rename en_ij nflows_e
							rename in_ij nflows_i
							rename es_ij av_es
							rename is_ij av_is
							
														

							foreach j in A_e A_i {
								if `alog' == 1 {
									replace `j' = ln(`j')
								}	
								if `anorm' == 1 {
									egen temp1 = mean(`j'), by(year)
									egen temp2 = sd(`j'), by(year)
									replace `j' = (`j' - temp1)/temp2
									drop temp*
								}	
							}
							
							if `af' == 0 {
									egen A_f = rowmean(A_e A_i)
									
							}
							if `af' == 1 {
									pca A_e A_i
									predict A_f
							}
							if `anorm' == 1 {
								loc j A_f
								egen temp1 = mean(`j'), by(year)
								egen temp2 = sd(`j'), by(year)
								replace `j' = (`j' - temp1)/temp2
								drop temp*
							}

							gsort -A_f
							format A* %10.3fc
							noi list year iso A_e A_i A_f  if _n<=10	
						//----------------
						save temp_r.dta, replace 
						 
						use  temp_accuracy.dta
						rename exporter iso 
						
						merge m:1 year iso using temp_r.dta, keepusing(A_e A_i sigmas) keep(3 1) nogen 
						
						rename iso exporter 
						rename A_e exporter_A_e
						rename A_i exporter_A_i
						rename sigmas exporter_sigmas
						rename importer iso 
						
						merge m:1 year iso using temp_r.dta, keepusing(A_e A_i sigmas) keep(3 1) nogen 
						
						rename iso importer
						rename A_e importer_A_e
						rename A_i importer_A_i 
						rename sigmas importer_sigmas
						sort year exporter importer
						erase temp_r.dta
						erase temp_accuracy.dta
						 
						*-------------------------------------------------------------------------------
						drop if exporter==importer
						recast float exportvalue_fob importvalue_fob, force
						*-----------------------------------------------------------------------------------------------
						
						egen tag_e = tag(exporter)
						egen tag_i = tag(importer)
						
						loc ae exporter_A_e // `ae' exporter_A_e exporter_sigmas
						loc ai importer_A_i // `ai' importer_A_i importer_sigmas
						
						*-------------------------------------------------------------------------------
						// Please double check why we get random small numbers?
						foreach i in importvalue_fob exportvalue_fob {
								replace `i' = . if `i'<10^3
						}
						*-------------------------------------------------------------------------------							
						// calculating weights
						cap drop w_*

						
						//qui sum exporter_A_e if tag_e, d
						qui sum `ae' if tag_e, d
						loc exp_90 = round( r(p90) , 0.001)
						loc exp_75 = round( r(p75) , 0.001)
						loc exp_50 = round( r(p50) , 0.001)
						loc exp_25 = round( r(p25) , 0.001)
						loc exp_10  = round( r(p10)  , 0.001)
						//gen w_e_0 = (exporter_A_e!=. & exporter_A_e > r(p5) )
						
						*
						//qui sum importer_A_i if tag_i, d
						qui sum `ai' if tag_i, d
						loc imp_90 = round( r(p90) , 0.001)
						loc imp_75 = round( r(p75) , 0.001)
						loc imp_50 = round( r(p50) , 0.001)
						loc imp_25 = round( r(p25) , 0.001)
						loc imp_10  = round( r(p10)  , 0.001)
						//gen w_i_0 = (importer_A_i!=. & importer_A_i > r(p5) )
						 
						//drop if importer=="ANS" & exporter_A_e<`exp_50'
						//drop if exporter=="ANS" & importer_A_i<`imp_50'
						
						gen w_e = exp(`ae') /( exp(`ae') + exp(`ai') )
						
						
						no di " ::  exporter A_e :: 10, 25, 50, 75, 95 = " `exp_10' " & " `exp_25' " & " `exp_50' " & " `exp_75' " & " `exp_90'
						no di " ::  importer_A_i :: 10, 25, 50, 75, 95 = " `imp_10' " & " `imp_25' " & " `imp_50' " & " `imp_75' " & " `imp_90'
						 
						
						// countries to include 
						gen w_e_0 = (`ae'!=. & `ae' > `exp_10' ) 
						gen w_i_0 = (`ai'!=. & `ai' > `imp_10' )

						
						noi di "	:: Estimating total trade flows between countries " 
						gen discrep = exp(abs(log(exportvalue_fob/importvalue_fob)))
							replace discrep = 99 if discrep==. 
						
						format cif_ratio `ae' `ai' importer_A_e w_e discrep %3.2fc
						
					
						*-----------------------------------------------------------------------------------------------
						gen float est value = .
							****
							format exportvalue_fob importvalue_fob estvalue  %20.0fc
							cap order year exporter importer exportvalue_fob importvalue_fob estvalue `ae' `ai' w_e discrep
							****
							loc res (`ae' !=. & `ai' !=. )
							noi replace estvalue = exportvalue_fob * w_e  + importvalue_fob * (1-w_e) ///
							        if `res' & exportvalue_fob!=. & importvalue_fob!=. // & discrep < 1.3 
							//br if estvalue==. & exportvalue_fob!=. & importvalue_fob!=.			
							*
							noi replace estvalue = importvalue_fob if `ae' < `exp_50' & `ai' >= `imp_90' & `res' & estvalue==.
							noi replace estvalue = exportvalue_fob if `ae' >= `exp_90' & `ai' < `imp_50' & `res' & estvalue==.
							*
							noi replace estvalue = importvalue_fob if `ae' < `exp_25' & `ai' >= `imp_75' & `res' & estvalue==.
							noi replace estvalue = exportvalue_fob if `ae' >= `exp_75' & `ai' < `imp_25' & `res' & estvalue==.
						
							*
							replace estvalue = max(importvalue_fob , exportvalue_fob) if `res' & w_e_0==1 & w_i_0==1  & estvalue==.
							replace estvalue = importvalue_fob if w_i_0==1 & estvalue==. 
							replace estvalue = exportvalue_fob if w_e_0==1 & estvalue==. 
							replace estvalue = importvalue_fob if estvalue==.
				
							* 
							replace estvalue = .  if estvalue==0
							//replace estvalue =0 if exporter_A_e<`exp_ub' & importer=="nan"
						*-----------------------------------------------------------------------------------------------	
						
						drop discrep 
						compress
						format exportvalue_fob importvalue_fob estvalue %15.0fc
						format cif_ratio w_e %9.3fc
						/*
						egen total_value = sum(estvalue), by(year exporter)
						gen share_exporter = estvalue / total_value
						format share_exporter %9.3fc
						loc res share_exporter > 0.75  & share_exporter!=. ///
							& total_value>10^7 & `ai'<`imp_50' & `ae'==. ///
							& (exporter!="BRN" & importer!="MYS") ///
							& (exporter!="DJI" & importer!="SAU")
						sum estvalue if `res'
						loc flows2drop  = r(mean) + 0 
						if `flows2drop' > 0 & `flows2drop'!=. {
							noi di " - Flows to be dropped " 
							noi list year exporter importer  estvalue share_exporter if `res'
							replace estvalue =. if `res'
						}
						drop if estvalue == .
						*/
						egen mintrade = rowmin(exportvalue_fob importvalue_fob)
						replace estvalue = mintrade if mintrade!=. & estvalue==. 
					
						
						order year exporter importer exportvalue_fob importvalue_fob estvalue cif_ratio w_*  `ae' `ai'
						keep year exporter importer exportvalue_fob importvalue_fob estvalue cif_ratio w_*  `ae' `ai'
						
						rename exportvalue_fob value_exporter
						rename importvalue_fob value_importer 
						rename estvalue value_final

						
						save "${path}/Weights/weights_`y'.dta", replace
						noi di "	:: finished weights and totals"
						noi di "	"
				}		
}	
//------------------------------------------------------------------------------				 

//use "${path}/Weights/weights_2008.dta", clear
 

//------------------------------------------------------------------------------				 
clear 
forval y = `start'/`end' {
	append using "${path}/Weights/weights_`y'.dta" 
}
*keep year exporter importer value_exporter value_importer value_final `ae' `ai'

*------------------------------------------------------------------------------
save "${path}/Weights/Totals_final.dta", replace			
*-------------------------------------------------------------------------------	
/*	
	
*-------------------------------------------------------------------------------
use "Weights/Totals_final.dta", clear
levelsof year, local(lyears)
clear 
foreach y of local lyears {
	quietly {
		use "Weights/Totals_final.dta"
		keep if year==`y'
		cap drop year
		sort exporter importer
		merge 1:1 exporter importer using "Weights/weights_`y'.dta", nogen keep(3 1)
		replace estvalue = value_final		
		save "Weights/weights_`y'.dta", replace  
	}	
}
*-------------------------------------------------------------------------------
*/


