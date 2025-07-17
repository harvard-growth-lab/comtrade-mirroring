*===============================================================================
clear all
set more off
set matsize 10000
*===============================================================================

*----------------------------------------------------------------------
*global path /Users/sbustos/Documents/_s/_datos/Trade/Trade_data/
*global path /nfs/projects_nobackup/c/cidgrowlab/SBL/Trade_data
//global path "D:/Data/Trade_data"
//global path /Users/seb335/Dropbox/datasets/Trade_data

global path /n/hausmann_lab/lab/atlas/bustos_yildirim/stata_files/
//global path "/nfs/projects_nobackup/c/cid_trade/Trade_data"
*----------------------------------------------------------------------
cd "$path"

// calculate RCA, diversity, ubiquity, complexity (inverting matrices)
// Calcualte the product space 
// non standard:
// calculate indicators for countries, limited number of countries 
// that we view as reliable 
// but we are calculating indicators for all the countries, 
// from the most reliable countries
// reduce the noise
// use those complexity metrics to compute the data for the excluded countries 

global classification H4 // S2_final // H0 // S2_final
global initial = 2012
global end = 2021
*/

global pathccpyfiles ${path}/update/Final/${classification}ccpy
//global pathccpyfiles ${path}/update/Final/${classification}
global supportfiles ${path}/update/Support_Files
global complexity ${path}/update/Final/Complexity

/// /Users/seb335/Dropbox/datasets/Trade_data/update/Final/H0ccpy

forvalues year = $initial/$end {
		quietly {
			mata mata clear
			//cap use "${classification}/${classification}_`year'.dta", clear
			use "${pathccpyfiles}/${classification}_`year'.dta", clear
			*noi tab yearstop
			tempfile fulldata selecteddata imports
			keep if year == `year'	
			noi di " " 
			noi di " :: Doing year " `year'
			
			*===============================================================================
			cap rename value_final export_value
			keep year exporter importer commoditycode export_value 
			*===============================================================================

			cap replace commoditycode = substr(commoditycode, 1, 4) if length(commoditycode) >= 5
			//cap destring commoditycode, replace force 
			cap drop if commoditycode==.
			drop if commoditycode=="" | commoditycode==" "
			cap drop if commoditycode=="."
			recast float export_value, force
			// check if data is complete 
			collapse  (sum)  export_value , by(year exporter importer commoditycode)
			recast float export_value, force
			
			preserve
				collapse  (sum) import_value = export_value , by(year importer commoditycode)
				recast float import_value, force
				rename importer exporter
				save `imports', replace 
			restore 
			
			collapse  (sum) export_value , by(year exporter commoditycode)
			recast float export_value, force
			
			merge 1:1 year exporter commoditycode using `imports', nogen 
			
			fillin year exporter commoditycode
			keep year exporter commoditycode export_value import_value
			replace export_value = 0 if export_value==.
			replace import_value = 0 if import_value == . 
			
			merge m:1 exporter year using "${supportfiles}/auxiliary_statistics" ///
			          , keep(3 1) nogenerate keepusing(population gdp_pc)
			
			replace population = 0 if population == .
			merge m:1 exporter using "${supportfiles}/obs_atlas", keep(3 1) keepusing(exporter)
			gen byte inatlas=(_merge==3)
			drop _merge 
			//cap replace inatlas = 1 if inlist(exporter,"SDN","SYR","HKG")
			replace inatlas =0 if inlist(exporter,"SYR","HKG","GNQ")
			replace inatlas =1 if inlist(exporter,"ARM","BHR","CYP","MMR","SWZ","TGO","BFA")
			replace inatlas =1 if inlist(exporter,"COD","LBR","SDN","SGP")
			
			sort exporter commoditycode 				
			egen tot_by_country = total(export_value), by(exporter)
			egen tot_by_prod = total(export_value), by(commoditycode)
			drop if tot_by_country==0 | tot_by_prod==0
			drop tot_by*
			compress
			noi di " :: ---- main file"
			save `fulldata', replace 
			*------------------------------------------------------------------------------------------
			
			keep if inatlas
			
			
			collapse  (sum) export_value (first) population gdp_pc, by(exporter commoditycode )	
			recast float export_value population gdp_pc, force
			
			if "${classification}"=="S2_final" { 
				drop if inlist(commoditycode,"9310","9610","9710","9999","XXXX")
			}
			if ("${classification}"=="H0") | ("${classification}"=="H4") { 
				drop if inlist(commoditycode,"7108","9999","XXXX")
			}	
			
			*----------------------------------------------------------------------------
			preserve
					egen temp1 = total(export_value), by(  commoditycode)
					egen temp2 = total(export_value) 
					egen temp3 = total(export_value), by( exporter)
					gen rca = (export_value / temp3) / (temp1/temp2)
					gen mcp = (rca>=1)
			
					egen HH = total(export_value), by(  commoditycode)
					replace HH = (export_value / HH)^2
					collapse  (sum) export_value HH mcp, by(commoditycode)
					//cap drop if export_value<1
					//cap drop if export_value==0
					//cap drop if export_value==.
					sort export_value
					egen share = total(export_value)
					replace share = 100*(export_value / share)
					gen cumshare = sum(share)
					
					gen ene = 1/HH // effective number of exporters 
					sort cumshare
					cap drop c1 c2 c3 call
					//
					gen byte c1 = (cumshare<=0.025)
					gen byte c2 = (ene<=2)
					gen byte c3 = (mcp<=2)
					egen call = rowtotal(c1 c2 c3)
					replace call = (call>0)
					replace call = 1 if export_value<1 
					//noi tabstat c1 c2 c3  call, s(sum)
					distinct commoditycode
					loc ni = r(ndistinct)
					qui sum call
					noi di " :: ---- number of products to exclude = " r(sum) " of " `ni'
					qui sum share if call==0
					noi di " :: ---- share to include = " r(sum)
					levelsof commoditycode if call==1, local(l2drop)
			restore
			*----------------------------------------------------------------------------
			
			*----------------------------------------------------------------------------
			noi di " :: ---- dropping least traded products " 
			foreach j of local l2drop {
				qui drop if commoditycode == "`j'"
			}	
			// uses stata package, use Shreyas's package here?
			ecomplexity export_value, i(exporter) p(commoditycode)			
			noi di " :: ---- complexity of selected countries and products"
			
			sort exporter commoditycode 
			loc income2use gdp_pc
			putmata `income2use', replace
			mata `income2use' = rowshape(`income2use', Nix) // rows(RCA)
			
			mata RCA = editmissing(RCA,0)
			mata prody = (RCA:/colsum(RCA)) :* `income2use'  
			mata prody = colsum(prody)
			
			mata pci1 = kp1d
			mata rca1 = RCA
			mata eci1 = (rca1:>=1):* pci1'
			mata eci1 = eci1 :/ rowsum( (rca1:>=1))
			mata eci1 = J(rows(rca1),cols(rca1),1) :* eci1
			rename rca rca1
			rename pci pci1
			rename eci eci1
			rename density density1 
			egen byte tagp = tag(commoditycode)
			qui sum pci1 if tagp
			replace pci1 = (pci1 - r(mean))/r(sd)
			drop tagp
			//gdistinct commoditycode			
			save  `selecteddata', replace 
			qui levelsof commoditycode, local(l2keep)
			*----------------------------------------------------------------------------
			
			
			*----------------------------------------------------------------------------
			noi di " :: ---- complexity with all countries, dropping least traded products"
					use `fulldata', clear
					keep exporter commoditycode export_value 
					gen byte tokeep = 0
					//gen byte tokeep = 1
					foreach j of local l2keep {
					//foreach j of local l2drop {
						qui replace tokeep=1  if commoditycode == "`j'"
						//qui replace tokeep=0  if commoditycode == "`j'"
					}
					
					//gdistinct commoditycode if tokeep
					keep if tokeep 
					fillin exporter commoditycode
					keep exporter commoditycode export_value 
					replace export_value = 0 if export_value==.
					 
					distinct commoditycode
					loc ni = r(ndistinct)
					sort exporter commoditycode
					putmata export_value, replace 
					mata export_value = colshape(export_value,`ni')
					mata rca2 = (export_value :/ rowsum(export_value)) :/ (colsum(export_value):/sum(export_value))
					
					mata mcp = (rca2:>=1.0)
					mata mata des export_value rca2 mcp pci1
					// taking complexity, forcing smaller group (trusted) pcis, 
					// into all country pci
					mata eci2 = mcp :* pci1'
					mata eci2 = rowsum(eci2) :/ rowsum(mcp)
					mata eci2 = J(rows(eci2),rows(kp1d),1) :* eci2
					mata expy = rowsum((export_value:/rowsum(export_value))  :* prody)
					mata expy = J(rows(export_value),cols(export_value),1) :* expy
					mata prody = J(rows(export_value),cols(export_value),1) :* prody
		 
		
					
					mata eci2 = colshape(eci2,1)
					mata rca2 = colshape(rca2,1)
					mata expy = colshape(expy,1)
					mata prody = colshape(prody, 1)					

					getmata eci2 rca2 expy prody, replace 
					mata rca2 = colshape(rca2,`ni')
					mata eci2 = colshape(eci2,cols(rca2))
					merge 1:1 exporter commoditycode using `selecteddata' ///
					          , keepusing(eci* rca* pci* coi* cog* density*) nogen
					egen byte tagc = tag(exporter)
					sum eci2 if tagc
					replace eci2 = (eci2-r(mean))/r(sd)
			
			save `selecteddata', replace 
			*----------------------------------------------------------------------------
			

			*----------------------------------------------------------------------------
			use `fulldata', clear
			noi di " :: ---- complexity with all countries, all products"
			fillin exporter commoditycode
			keep exporter commoditycode export_value `income2use'  import_value
			replace export_value = 0 if export_value==.
			replace import_value = 0 if import_value == .
			distinct commoditycode
			loc ni = r(ndistinct)
			di `ni'
			sort exporter commoditycode
			putmata export_value `income2use', replace 
			mata export_value = colshape(export_value,`ni')
			mata `income2use' = colshape(`income2use',`ni')
			mata export_value[.,`ni'] = J(rows(export_value),1,0)
			mata rca3 = (export_value :/ rowsum(export_value)) :/ (colsum(export_value):/sum(export_value))
			//
			mata mcp3 = (rca3:>=1)
			
			
			mata mata des rca*
			mata mata des eci*
			
			mata pci3 = (rca3:>=1) :* eci2[.,1]
			mata pci3 = colsum(pci3)'
			mata pci3 = pci3 :/ colsum(rca3:>=1)'
			//mata mata des pci*
			mata pci3 = J(rows(rca3),cols(rca3),1) :* pci3'
			mata prody3 = (rca3:/colsum(rca3)) :* `income2use'
			mata prody3 = colsum(prody3)
			mata prody3 = J(rows(export_value),cols(export_value),1) :* prody3
			
			
			*------------------------------------------------------------------------------
			noi di " :: ---- product space, all countries, all products"			
			mata M = (rca3:>=1) 
			mata C = M'*M
			mata Nps = cols(M)
			mata Ncs = rows(M)
			mata S = J(Nps,Ncs,1)*M
			mata P1 = C:/S
			mata P2 = C:/S'  
			// who is nearby which product?
			mata proximity = (P1+P2 - abs(P1-P2))/2 - I(Nps)
			mata density3 = proximity' :/ (J(Nps,Nps,1) * proximity')
			mata density3 = M * density3
			//mata coi = ((density3 :*(1 :- M)):* ( J(Ncs,1,1)*pci3[1,.] )  )*J(Nps,Nps,1)
			mata opportunity_value =  ((density3:*(1 :- M)):*pci3)*J(Nps,Nps,1)
            
			mata opportunity_gain = (J(Ncs,Nps,1) - M ):*((J(Ncs,Nps,1) - M ) * (proximity :* ((pci3[1,.]':/(proximity*J(Nps,1,1)))*J(1,Nps,1))))
			
			//mata mata des M density3 opportunity_value  opportunity_gain
			 
			
			local vars2stata rca3 pci3 M density3 prody3 opportunity_value opportunity_gain 
			foreach j of local vars2stata {
				mata `j' = colshape(`j',1)
			}
			
			getmata  `vars2stata' , replace
			
			noi di " :: ---- combining files"
			merge 1:1 exporter commoditycode using `selecteddata' ///
			      , keepusing(eci* rca* pci* coi* cog* density* expy prody ) nogen
			
			foreach j in eci1 eci2 expy {
				cap drop temp
				egen temp = mean(`j'), by(exporter)
				replace `j' = temp if `j'==.
				cap drop temp
			}	
			
			egen byte tagc = tag(exporter)
			egen byte tagp = tag(commoditycode)
			
			order exporter commodity export_value rca* pci* eci*
			
			qui sum pci3 if tagp
			replace pci3 = (pci3 - r(mean))/r(sd) if pci3!=. 
			
			qui sum opportunity_value if tagc
			replace opportunity_value = (opportunity_value-r(mean))/r(sd) if opportunity_value!=. 
			
			*---------------------------------------------------------------------
			noi di " :: ---- combining variables"
			//stop
			//br if inlist(commoditycode,"9310","9610","9710","9999","XXXX")
			gen rca = rca1
			replace rca =  rca2 if rca==.
			replace rca =  rca3 if rca==.
			drop rca1 rca2 rca3
			
			gen eci = eci1
			replace eci = eci2 if eci==.
			drop eci1 eci2
			
			gen pci = pci1
			replace pci = pci3 if pci==.
			drop pci1 pci3
			
			replace prody = prody3 if prody==.
			drop prody3
			
			
			gen density = density1
			replace density = density3 if density==. 
			cap drop density1 density3
			
			gen oppval = coi
			replace oppval = opportunity_value if oppval==. 
			gen oppgain = cog 
			replace oppgain = opportunity_gain if oppgain==. 
			cap drop coi cog opportunity_value opportunity_gain 
			
			rename M mcp 
			cap gen distance = 1 - density
			cap drop density
			
			*------------------------------------------------------------------------------
			noi di " :: ---- cleaning, formating and labeling "
			
			merge m:1 exporter using "${supportfiles}/obs_atlas", keepusing(exporter) keep(3 1)
			gen byte inatlas=(_merge==3)
			replace inatlas =0 if inlist(exporter,"SYR","HKG","GNQ")
			replace inatlas =1 if inlist(exporter,"ARM","BHR","CYP","MMR","SWZ","TGO","BFA")
			replace inatlas =1 if inlist(exporter,"COD","LBR","SDN","SGP","PAN")
			cap drop _merge
			
			cap egen temp = total( export_value), by(exporter)
			cap drop if temp==0
			cap drop temp
			
			*------------------------------------------------------------------------------
			/*
			if "${classification}"=="S2_final" { 
				foreach v in mcp rca eci pci oppval oppgain distance {
					replace `v' = . if inlist(commoditycode,"9310","9610","9710","9999","XXXX")
				}
			}
			if "${classification}"=="H0" { 
				foreach v in mcp rca eci pci oppval oppgain distance {
					replace `v' = . if inlist(commoditycode,"7108","9999","XXXX")
				}
			}
			*/
			
			foreach v in export_value mcp rca eci pci oppval oppgain distance prody expy {
					replace `v' = 0 if `v'==.
			}
			*------------------------------------------------------------------------------
			
			drop tag*
			*------------------------------------------------------------------------------
			* Labeling
			*------------------------------------------------------------------------------
			cap label var year "Year"
			cap label var exporter "Exporter ISO code"
			cap label var export_value  "Exports value - estimate"
			cap label var import_value  "Imports value - estimate"
			cap label var rca "Revealed Comparative Advantage (RCA)"
			cap label var rpop "Revealed per capita Advantage (RPOP)"
			cap label var density "Product Density "
			cap label var distance "Product Distance "
			cap label var coi "Complexity Outlook Index"
			cap label var oppval  "Complexity Outlook Index (f. oppval)"
			cap label var cog "Opportunity Gain"
			cap label var oppgain "Complexity Opportunity Gain (f. Oppgain)"
			cap label var eci "Economic Complexity Index" 
			cap label var pci "Product Complexity Index"
			cap label var diversity "Country Diversity"
			cap label var ubiquity "Product Ubiquity"
			cap label var piet_c "Country Measure of Pietronero"
			cap label var piet_p "Product Measure of Pietronero"
			cap label var mcp "Presence (M=1 if RCA>1)"
			cap label var inatlas "1 = list of ranking/atlas countries"
			cap label var prody "PRODY - Product income"
			cap label var expy "EXPY - Product income"
			*------------------------------------------------------------------------------
			
			sort  exporter commoditycode
			format rca dist  oppval oppgain eci pci   %9.2fc
			format prody expy %9.0fc
			
			format export_value import_value  %20.0fc 
			format inatlas   %1.0fc
			
			gen int year = `year'
			noi di " :: ---- saving file"
			
			order year exporter commoditycode inatlas export_value ///
				import_value rca mcp eci pci oppval oppgain distance prody expy
			compress 
			
			saveold "${complexity}/Complexity_${classification}_`year'", replace  version(12)
			
		}	
			
}
*===============================================================================

  

mata mata clear 

//==============================================================================
// 		country and product complexity files 
//==============================================================================


//global initial = 2019
//global end = 2021
*/

cd "$path/"

clear 
forvalues year = $initial/$end {
	quietly {
		append using "${complexity}/Complexity_${classification}_`year'"
		cap drop __00*	
	}
}

cap  drop __00*
saveold "${complexity}/${classification}_cpy_all", replace   version(12)
//saveold "D:/Dropbox/datasets/Trade_data/${classification}_cpy_all", replace version(12)

//zipfile "Complexity/${classification}_cpy_all.dta", saving("D:/Dropbox/datasets/Trade_data/${classification}_cpy_all", replace)

use "${complexity}/${classification}_cpy_all", clear
cap  drop __00*

egen total_exports = sum(export_value), by(year exporter)
drop if exporter=="ANS"
egen byte t_yc=tag(year exporter)
keep if t_yc==1

*----------------------------------------------------------------------------
merge 1:1 exporter year using "${supportfiles}/auxiliary_statistics", keep(3 1) nogenerate keepusing(population gdp_pc)
			replace inatlas =0 if inlist(exporter,"SYR","HKG","GNQ")
			replace inatlas =1 if inlist(exporter,"ARM","BHR","CYP","MMR","SWZ","TGO","BFA")
			replace inatlas =1 if inlist(exporter,"COD","LBR","SDN","SGP","PAN")
			
			// SGP
*----------------------------------------------------------------------------
			
keep year exporter eci oppval expy total_exports population gdp_pc inatlas 
order year exporter total_exports eci oppval expy  population gdp_pc inatlas 

egen rank_eci_all=rank(-eci), by(year) unique
egen rank_eci_atlas=rank(-eci) if inatlas==1, by(year) unique

sort year exporter
cap label var total_exports "Total exports in current dollars"
cap label var inatlas "Selected for Atlas"
cap label var exporter "Exporter ISO code"

compress

format eci oppval %9.2fc
format expy %9.0fc
format total_exports population %20.0fc 
format inatlas rank_eci_all rank_eci_atlas %1.0fc

saveold "${complexity}/Country_complexity_${classification}", replace  version(12)
//=========================================================================================================


use "${complexity}/${classification}_cpy_all", clear
	replace mcp = mcp * inatlas
	keep year commoditycode pci prody export_value  mcp inatlas
	collapse  (first) pci prody (sum) total_exports = export_value (sum) mcp inatlas, by(year commoditycode)   
	gen ubiquityshare = mcp/inatlas 
	drop  mcp inatlas
	compress
	*------------------------------------------------------------------------------
	cap label var year "Year"
	cap label var total_exports  "Total exports value"
	cap label var pci "Product Complexity Index"
	cap label var prody "PRODY - product income "
	cap label var ubiquity "Product Ubiquity share (% of countries with presence)"
	*------------------------------------------------------------------------------
	format pci ubiquity %9.2fc
	format prody %9.0fc
	format total_exports  %20.0fc 

saveold "${complexity}/Product_complexity_${classification}", replace  version(12)
//=========================================================================================================


