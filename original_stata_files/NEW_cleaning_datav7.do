*-------------------------------------------------------------------------------
cls
clear all 
set more off 
*-------------------------------------------------------------------------------
//global path "D:/Data/Trade_data"
//global path /Users/sbustos/Dropbox/datasets/Trade_data/
//global path "/nfs/projects_nobackup/c/cid_trade/Trade_data"

cd "${path}/"


loc syear ${initial}
loc eyear ${end}

*===============================================================================
mata mata set matastrict off
cap mata: mata drop loadingMs()
cap mata: mata drop reweightMs()
*-------------------------------------------------------------------------------
// The following function is used to populate matrices in a fast way //
mata:
	real matrix function loadingMs(real colvector idc, real colvector idp, real colvector val, Npair, Nprod)
	
	{
	    M = J(Npair, Nprod, 0)
		
		for (i=1; i<=rows(idc); i++) {
			r = idc[i]
			c = idp[i]
			M[r,c] = val[i]
		}
		M = editmissing(M,0)
		return(M)
		
	}
end	
*-------------------------------------------------------------------------------

*-------------------------------------------------------------------------------
mata:
	real matrix function reweightMs(real colvector VF, real colvector value_final, Nprod)
	
	{
		VF = colshape(VF,Nprod)
		sumVF = rowsum(VF)
			
		temp1 = (((value_final :/ sumVF):>1.20) :+ ((value_final :- sumVF):>2.5*10^7) :+ (value_final:>10^8)):==3
		temp2 = ((value_final :> 10^8) :+ (sumVF:<10^5)):==2
		xxxx = (temp1:+temp2):>0
		
		value_xxxx = (value_final :- sumVF) :* (xxxx:==1)
		value_reweight = value_final :- value_xxxx
					
		VR = VF :- VF:*(VF:<1000)
		VR = VR :/ rowsum(VR)
		VR = VR :* value_reweight
					
		VR = editmissing(VR,0)
		VR = VR , value_xxxx
		
		return(VR)				
				 
	}
end	
*===================================================================================


			
			
*===================================================================================
forval y=`syear'/`eyear' { 
	quietly {
			clear
			no di "Doing year = " `y'	
			*-------------------------------------------------------------------------------
			timer on 1
			noi di "		- 1 : Importing and loading zip file with Python"
		 
			*-------------------------------------------------------------------------------
			cd "$path/${classification}"
			clear
			noi di "			:: Extracting data with Python"
			cap erase temp.csv
			cap erase temp.dta
			// opens compresses files and retrives the useful columns in Python
			loc program  ${path}/DO_files/comtrade_reads_zip.py
			loc rfile  ${path}/${classification}_raw/${classification}_`y'.zip
			!python3 `program' `rfile'
			//shell python `program' `rfile'  
			//shell python36 `program' `rfile' 	 
			//!python `program' `rfile' 
			 
			timer off 1
			*-------------------------------------------------------------------------------
			/*
			clear 
			tempfile comtradefile
			cap use temp.dta, clear
			cap use update/temp.dta, clear
			keep reporter_iso partner_iso commoditycode agglevel tradevalue tradeflow  
			recast float tradevalue, force
			*/
			
			
			*-------------------------------------------------------------------------------
			timer on 2						
			noi di "		- 2 : Creating countrypairs and id of products"
			*-------------------------------------------------------------------------------
			clear 
			tempfile idproducts
			noi di "				:: Loading to Stata"
			cap use commoditycode agglevel tradevalue using temp.dta, clear
			cap use commoditycode agglevel tradevalue using update/temp.dta, clear
			
			recast float tradevalue, force 
			
			
			if "${classification}"=="H0" {
				*-------------------------------------------------------------------------------
				// file with all products at 6d
				keep if agglevel==6
				replace commoditycode = "00"+commoditycode if length(commoditycode)==4
				replace commoditycode = "0"+commoditycode  if length(commoditycode)==5
				cap replace commoditycode="999999" if  substr(commoditycode,1,4)=="9999"
				*-------------------------------------------------------------------------------
			}
			
			if ("${classification}"=="S1") | ("${classification}"=="S2") {
				*-------------------------------------------------------------------------------
				// file with all products at 4d
				keep if agglevel==4
				replace commoditycode = "00"+commoditycode if length(commoditycode)==2
				replace commoditycode = "0"+commoditycode  if length(commoditycode)==3
				*-------------------------------------------------------------------------------
			}
			gcollapse (sum) tradevalue, by(commoditycode)
			recast float tradevalue, force
			drop if tradevalue<=10^5
			
			
			egen int idprod = group(commoditycode)
			keep commoditycode idprod
			compress
			//keep if idprod<=1000
			sort idprod
			putmata idprod, replace 
			qui sum idprod
			global Nidprod = r(max)
			save `idproducts'

			*-------------------------------------------------------------------------------

			*-------------------------------------------------------------------------------
			
			tempfile countrypairs
			cap use "${path}/Weights/weights_`y'.dta", clear
			cap use "/Users/sbustos/Dropbox/datasets/Trade_data/Weights/weights_`y'.dta", clear
			drop if value_final<10^5
			compress
			egen int idpair = group(exporter importer)
			//keep if idpair<=5000
			sort idpair
			putmata value_final cif_ratio w_e w_e_0 w_i_0, replace 
			keep exporter importer idpair
			putmata idpair, replace
			qui sum idpair
			global Nidpair = r(max)
			save `countrypairs'
			timer off 2
			*-------------------------------------------------------------------------------
			
			*-------------------------------------------------------------------------------
			timer on 3
			noi di "		- 3 : Loading trade data ccpy"
			*-------------------------------------------------------------------------------
			
			clear 
			tempfile comtradefile
			noi di "				:: Loading to Stata"
			cap use temp.dta, clear
			cap use update/temp.dta, clear
			keep reporter_iso partner_iso commoditycode agglevel tradevalue tradeflow  
			
			if "${classification}"=="H0" {
				*-------------------------------------------------------------------------------
				keep if agglevel==6
				drop agglevel
				recast float tradevalue, force
				replace commoditycode = "00"+commoditycode if length(commoditycode)==4
				replace commoditycode = "0"+commoditycode  if length(commoditycode)==5
				*-------------------------------------------------------------------------------
			}
			
			if ("${classification}"=="S1") | ("${classification}"=="S2") {
				*-------------------------------------------------------------------------------
				keep if agglevel==4
				drop agglevel
				recast float tradevalue, force
				replace commoditycode = "00"+commoditycode if length(commoditycode)==2
				replace commoditycode = "0"+commoditycode  if length(commoditycode)==3
				*-------------------------------------------------------------------------------
			}			
			compress
			
			*-------------------------------------------------------------------------------				
			// Dealing with Germany (reunification) and Russia
			*-------------------------------------------------------------------------------
			drop if reporter_iso == "DEU" & partner_iso=="DDR"
			drop if reporter_iso == "DDR" & partner_iso=="DEU"
			replace partner_iso  = "DEU" if inlist(partner_iso,"DEU","DDR")
			replace reporter_iso = "DEU" if inlist(reporter_iso,"DEU","DDR")
			
			replace partner_iso  = "RUS" if inlist(partner_iso,"RUS","SUN")
			replace reporter_iso = "RUS" if inlist(reporter_iso,"RUS","SUN")
			*-------------------------------------------------------------------------------		
			cap drop if commoditycode=="TOTAL"
			*-------------------------------------------------------------------------------
			
			foreach i in NAN nan WLD {
				cap drop if reporter_iso == "`i'"
				cap drop if partner_iso == "`i'"
			}	
						
			timer off 3
			*-------------------------------------------------------------------------------
			
			*-------------------------------------------------------------------------------
			timer on 4
			noi di "		- 4 : Loading trade data to Mata"
			*-------------------------------------------------------------------------------
				
			preserve
					keep if tradeflow == 2 // Exports 
					drop tradeflow
					rename reporter_iso exporter
					rename partner_iso importer 
					
					gcollapse (sum) tradevalue, by(exporter importer commoditycode)
					recast float tradevalue, force
					drop if tradevalue<1000
					
					merge m:1 exporter importer using `countrypairs', nogen keep(3)
					merge m:1 commoditycode using `idproducts', nogen keep(3)

					putmata idc=idpair  idp=idprod val=tradevalue, replace 
					
					mata Me=loadingMs(idc,idp,val, ${Nidpair},${Nidprod})
				    mata mata drop idc idp val   
				   		
			restore
			
			*-------------------------------------------------------------------------------
			keep if tradeflow == 1 // Imports 
			drop tradeflow
			rename reporter_iso importer
			rename partner_iso exporter
			gcollapse (sum) tradevalue, by(exporter importer commoditycode)
			recast float tradevalue, force 
			drop if tradevalue<1000
			
			merge m:1 exporter importer using `countrypairs', nogen keep(3)
			merge m:1 commoditycode using `idproducts', nogen keep(3)
			
			putmata idc=idpair idp=idprod val=tradevalue, replace 
			
			mata Mi = loadingMs(idc, idp, val, ${Nidpair}, ${Nidprod} )
			mata mata drop idc idp val
			
			timer off 4
			clear
			*-------------------------------------------------------------------------------
			
			//==============================================================================
			// STARTING MATA SECTION
			//==============================================================================
			
			*-------------------------------------------------------------------------------			
			noi di "		- 5 : Trade reconciliation"
			*-------------------------------------------------------------------------------
			timer on 5
			foreach v in cif_ratio w_e w_e_0 w_i_0 { 
				   mata `v' = editmissing(`v',0)
			}
			
			mata Mi = Mi:*(1:-cif_ratio)
			mata mata drop cif_ratio
			
			mata Me  = colshape(Me,1)
			mata Mi  = colshape(Mi,1)
			mata w_e = colshape(w_e,1)
					
			*-------------------------------------------------------------------------------
						
			*-------------------------------------------------------------------------------	
			mata trdata =  1:*(((Me:>0) :+ (Mi:>0)):>1) ///
							:+1:*((Me:>0)) ///
							:+2:*((Mi:>0)) 
			     
			mata accuracy  =  1:*(((w_e_0:>0) :+ (w_i_0:>0)):>1) ///
							  :+1:*((w_e_0:>0)) ///
							  :+2:*((w_i_0:>0)) 
			
			mata accuracy = J(${Nidpair}, ${Nidprod} ,1):*accuracy				 
			mata accuracy = colshape(accuracy,1)
			
			mata w_e = J(${Nidpair}, ${Nidprod} ,1):*w_e
			mata w_e = colshape(w_e,1)
			
			*-------------------------------------------------------------------------------
							 
			 
			*-------------------------------------------------------------------------------
			mata VF =  ((w_e:* Me) :+ ((1:-w_e):* Mi)) :* ((trdata:==4):*(accuracy:==4)) ///
					:+ (Mi :* ((trdata:==2):*(accuracy:==2))) :+ (Mi :* ((trdata:==2):*(accuracy:==4))) ///
					:+ (Me :* ((trdata:==1):*(accuracy:==1))) :+ (Me :* ((trdata:==1):*(accuracy:==4))) ///
					:+ (Mi :* ((trdata:==4):*(accuracy:==2))) ///
					:+ (Me :* ((trdata:==4):*(accuracy:==1))) ///
					:+ (0.5:*(Me:+Mi) :* ((trdata:==4):*(accuracy:==0))) ///
					:+ (Mi :* ((trdata:==2):*(accuracy:==0))) ///
					:+ (Me :* ((trdata:==1):*(accuracy:==0))) ///
					:+ (Mi :* ((trdata:==2):*(accuracy:==1))) ///
					:+ (Me :* ((trdata:==1):*(accuracy:==2))) 
			*-------------------------------------------------------------------------------		
			
			mata mata drop trdata accuracy 
			mata mata drop w_e w_e_0 w_i_0
			
			mata VR = reweightMs(VF, value_final, ${Nidprod})
			mata mata drop value_final
			mata mata drop VF
			
			*-------------------------------------------------------------------------------
			foreach v in Me Mi {
					mata `v' = colshape(`v',${Nidprod})
					mata `v' = `v', J(rows(`v'),1,0)
			}
			
			mata idprod = idprod \ (${Nidprod}+1)
			
			timer off 5
			*-------------------------------------------------------------------------------
			
			
			
			
			*-------------------------------------------------------------------------------
			// Exporting to stata
			*-------------------------------------------------------------------------------
			timer on 6
			noi di "		- 6 : Loading data to Stata from Mata NEW!"
			
			mata idpair = J(${Nidpair}, ${Nidprod}+1 ,1):*idpair
			mata idprod = J(${Nidpair}, ${Nidprod}+1 ,1):*idprod'
			
			foreach v in VR  Me Mi idpair idprod {
				mata `v' = colshape(`v',1) 
			}	
	
			*-------------------------------------------------------------------------------		
			mata results = idpair , idprod , Me , Mi , VR 
			mata mata drop idpair idprod  Me  Mi  VR 
			mata results = select(results, results[.,5]) // Dropping rows without data
			
			clear 
			getmata ( idpair idprod Me Mi VR ) = results 
			recast float Me Mi VR, force 
			timer off 6
			*-------------------------------------------------------------------------------
			
			
			*-------------------------------------------------------------------------------
			timer on 7
			noi di "		- 7 : Adding country names and product codes"			
			merge m:1 idprod using `idproducts', nogen keep(3 1)
			merge m:1 idpair using `countrypairs', nogen keep(3 1)
			drop idpair idprod
			
			if "${classification}"=="H0" {
					replace commoditycode="XXXXXX" if commoditycode==""
			}
			if ("${classification}"=="S1") | ("${classification}"=="S2") {
					replace commoditycode="XXXX" if commoditycode==""
			}
			
			preserve
				if "${classification}"=="H0" {
					gen val99 = VR if commoditycode=="999999" & importer=="ANS"
					replace VR = . if commoditycode=="999999" & importer=="ANS"
				}
				if ("${classification}"=="S1") | ("${classification}"=="S2") {
					gen val99 = VR if commoditycode=="9999" & importer=="ANS"
					replace VR = . if commoditycode=="9999" & importer=="ANS"
				}
				gcollapse (sum) val99 VR , by(exporter )
				gen ratio99 = val99 / VR
				//gsort -ratio99
				glevelsof exporter if ratio99>1/3, local(ctry2drop)
			restore
			*
			foreach c of local ctry2drop {
				if "${classification}"=="H0" { 
					drop if exporter=="`c'" &  commoditycode=="999999"
				}	
				if ("${classification}"=="S1") | ("${classification}"=="S2") {
					drop if exporter=="`c'" &  commoditycode=="9999"
				}
			}
			*
			format  VR  M* %15.0fc
			mata mata des
			///merge 1:1 exporter importer commoditycode using "/Users/sbustos/Dropbox/datasets/Trade_data/update/Final/H0/H0_2018", keepusing(export_value) nogen
			timer off 7
			*-------------------------------------------------------------------------------

			*-------------------------------------------------------------------------------
			noi di " "
			noi di " "
			noi di "------------------------------------------------------------------"
			noi di "- 1 : Importing and loading zip file"
			noi di "- 2 : Creating countrypairs and id of products"
			noi di "- 3 : Loading trade data ccpy"
			noi di "- 4 : Loading trade data to Mata"
			noi di "- 5 : Trade reconciliation"
			noi di "- 6 : Loading data to Stata from Mata"
			noi di "- 7 : Adding country names and product codes"
			noi di " "
			
			noi timer list 
			noi di "------------------------------------------------------------------"
			*-------------------------------------------------------------------------------

			
			rename VR value_final
			rename Me value_exporter
			rename Mi value_importer
			gen int year = `y'
			order year exporter importer commodity value_final
			sort exporter importer commodity
			cap label var value_final "Value - Final estimate"
			cap label var value_exporter "Value reported by exporter"
			cap label var value_importer "Value reported by importer"
			compress
			noi di "::	Saving dataset of year `y' "	
			 
			*-------------------------------------------------------------------------------
			save "${path}/${classification}/${classification}_`y'.dta", replace
			*------------------------------------------------------------------------------- 	
			noi di " "
			noi di " "
			
			cap erase "${path}/${classification}/temp.csv"
			cap erase "${path}/${classification}/temp.dta"
			
	}
}	
*===============================================================================================================			
			
