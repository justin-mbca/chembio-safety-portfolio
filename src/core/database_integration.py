#!/usr/bin/env python3
"""
Phase 1 Database Integration for ChemBio SafeGuard
Core safety databases with minimal overlap for enhanced threat detection

This module provides integration with:
- CWC Schedules: Chemical weapons and precursors
- CDC Select Agents: Restricted biological agents  
- DEA Controlled Substances: Drug precursors and controlled chemicals

Author: Xiangli (Justin) Zhang
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatabaseResult:
    """Standardized result from database lookup"""
    entity: str
    database: str
    risk_level: str
    confidence: float
    category: str
    restrictions: str
    regulatory_status: str
    explanation: str
    source_url: Optional[str] = None

class CWCScheduleDatabase:
    """
    Chemical Weapons Convention Schedule Database
    
    Maintains lists of chemicals restricted under international law.
    Schedule 1: Most dangerous, production prohibited except research
    Schedule 2: Precursors and key intermediates, limited production
    Schedule 3: Dual-use chemicals with legitimate uses, monitored
    """
    
    def __init__(self):
        self.last_updated = datetime.now()
        
        # Schedule 1: Chemical weapons and direct precursors (production prohibited)
        self.schedule_1_chemicals = {
            'sarin': {'cas': '107-44-8', 'formula': 'C4H10FO2P'},
            'soman': {'cas': '96-64-0', 'formula': 'C7H16FO2P'},
            'tabun': {'cas': '77-81-6', 'formula': 'C5H11N2O2P'},
            'vx nerve agent': {'cas': '50782-69-9', 'formula': 'C11H26NO2PS'},
            'vx': {'cas': '50782-69-9', 'formula': 'C11H26NO2PS'},
            'mustard gas': {'cas': '505-60-2', 'formula': 'C4H8Cl2S'},
            'bis(2-chloroethyl) sulfide': {'cas': '505-60-2', 'formula': 'C4H8Cl2S'},
            'lewisite 1': {'cas': '541-25-3', 'formula': 'C2H2AsCl3'},
            'lewisite 2': {'cas': '40334-69-8', 'formula': 'C4H4As2Cl4'},
            'lewisite 3': {'cas': '40334-70-1', 'formula': 'C6H6As3Cl3'},
            'ricin': {'cas': '9009-86-3', 'type': 'toxin'},
            'saxitoxin': {'cas': '35523-89-8', 'formula': 'C10H17N7O4'},
            'nitrogen mustard': {'cas': '51-75-2', 'formula': 'C5H11Cl2N'},
        }
        
        # Schedule 2: Key precursors (production limited and monitored)
        self.schedule_2_chemicals = {
            'thiodiglycol': {'cas': '111-48-8', 'formula': 'C4H10O2S'},
            'phosphorus oxychloride': {'cas': '10025-87-3', 'formula': 'POCl3'},
            'methylphosphonyl dichloride': {'cas': '676-97-1', 'formula': 'CH4Cl2OP'},
            'dimethyl methylphosphonate': {'cas': '756-79-6', 'formula': 'C3H9O3P'},
            'phosphorus trichloride': {'cas': '7719-12-2', 'formula': 'PCl3'},
            'trimethyl phosphite': {'cas': '121-45-9', 'formula': 'C3H9O3P'},
            'pinacolyl alcohol': {'cas': '464-07-3', 'formula': 'C6H14O'},
            '3,3-dimethylbutan-2-ol': {'cas': '464-07-3', 'formula': 'C6H14O'},
        }
        
        # Schedule 3: Dual-use chemicals (production monitored above thresholds)
        self.schedule_3_chemicals = {
            'phosgene': {'cas': '75-44-5', 'formula': 'CCl2O'},
            'hydrogen cyanide': {'cas': '74-90-8', 'formula': 'HCN'},
            'chloropicrin': {'cas': '76-06-2', 'formula': 'CCl3NO2'},
            'phosphorus pentasulfide': {'cas': '1314-80-3', 'formula': 'P2S5'},
            'triethanolamine': {'cas': '102-71-6', 'formula': 'C6H15NO3'},
            'diethyl ethylphosphonate': {'cas': '78-38-6', 'formula': 'C6H15O3P'},
            'diethyl phosphite': {'cas': '762-04-9', 'formula': 'C4H11O3P'},
        }
        
        # Alternative names and synonyms
        self.chemical_aliases = {
            'gb': 'sarin',
            'gd': 'soman', 
            'ga': 'tabun',
            'mustard': 'mustard gas',
            'yperite': 'mustard gas',
            'l1': 'lewisite 1',
            'l2': 'lewisite 2',
            'l3': 'lewisite 3',
            'poc13': 'phosphorus oxychloride',
            'cocl2': 'phosgene',
            'carbonyl chloride': 'phosgene',
        }
    
    def check_cwc_status(self, chemical_name: str) -> Optional[DatabaseResult]:
        """Check if a chemical is scheduled under the Chemical Weapons Convention"""
        
        if not chemical_name or len(chemical_name.strip()) < 2:
            return None
            
        # Normalize input
        chemical_lower = chemical_name.lower().strip()
        
        # Check aliases first
        if chemical_lower in self.chemical_aliases:
            chemical_lower = self.chemical_aliases[chemical_lower]
        
        # Check Schedule 1 (highest risk)
        for chem_name, details in self.schedule_1_chemicals.items():
            if self._chemical_matches(chemical_lower, chem_name):
                return DatabaseResult(
                    entity=chemical_name,
                    database="CWC Schedule 1",
                    risk_level="critical_risk",
                    confidence=0.95,
                    category="Chemical Weapon",
                    restrictions="Production prohibited except for research purposes (max 100g/year per facility)",
                    regulatory_status="CWC Schedule 1 - Mandatory declaration and inspection required",
                    explanation=f"Chemical weapon or direct precursor: {chem_name} (CAS: {details.get('cas', 'N/A')})",
                    source_url="https://www.opcw.org/chemical-weapons-convention"
                )
        
        # Check Schedule 2 (high risk)
        for chem_name, details in self.schedule_2_chemicals.items():
            if self._chemical_matches(chemical_lower, chem_name):
                return DatabaseResult(
                    entity=chemical_name,
                    database="CWC Schedule 2",
                    risk_level="high_risk",
                    confidence=0.85,
                    category="Chemical Weapon Precursor",
                    restrictions="Production limited and monitored (thresholds: 1kg research, 10kg industrial)",
                    regulatory_status="CWC Schedule 2 - Declaration and inspection rights apply",
                    explanation=f"Key precursor for chemical weapons: {chem_name} (CAS: {details.get('cas', 'N/A')})",
                    source_url="https://www.opcw.org/chemical-weapons-convention"
                )
        
        # Check Schedule 3 (medium risk)
        for chem_name, details in self.schedule_3_chemicals.items():
            if self._chemical_matches(chemical_lower, chem_name):
                return DatabaseResult(
                    entity=chemical_name,
                    database="CWC Schedule 3",
                    risk_level="medium_risk",
                    confidence=0.75,
                    category="Dual-Use Chemical",
                    restrictions="Production monitored above threshold quantities (30 tonnes/year)",
                    regulatory_status="CWC Schedule 3 - Annual declaration required above thresholds",
                    explanation=f"Dual-use chemical with weapons potential: {chem_name} (CAS: {details.get('cas', 'N/A')})",
                    source_url="https://www.opcw.org/chemical-weapons-convention"
                )
        
        return None
    
    def _chemical_matches(self, input_name: str, db_name: str) -> bool:
        """Check if chemical names match with fuzzy matching"""
        # Exact match
        if input_name == db_name:
            return True
        
        # Check if input contains the database name or vice versa
        if db_name in input_name or input_name in db_name:
            return True
        
        # Check individual words for complex chemical names
        input_words = set(input_name.split())
        db_words = set(db_name.split())
        
        # If significant overlap in words, consider it a match
        if len(input_words & db_words) >= min(len(input_words), len(db_words)) * 0.6:
            return True
        
        return False


class CDCSelectAgentsDatabase:
    """
    CDC Select Agents and Toxins Database
    
    Maintains the list of biological agents and toxins that have been determined
    to have the potential to pose a severe threat to public health and safety.
    Regulated under 42 CFR Part 73.
    """
    
    def __init__(self):
        self.last_updated = datetime.now()
        
        # HHS Select Agents and Toxins (42 CFR 73.3)
        self.hhs_select_agents = {
            # Bacteria
            'bacillus anthracis': {
                'type': 'bacteria',
                'disease': 'Anthrax',
                'bsl': '2/3',
                'notes': 'Spore-forming bacterium'
            },
            'brucella abortus': {'type': 'bacteria', 'disease': 'Brucellosis', 'bsl': '3'},
            'brucella melitensis': {'type': 'bacteria', 'disease': 'Brucellosis', 'bsl': '3'},
            'brucella suis': {'type': 'bacteria', 'disease': 'Brucellosis', 'bsl': '3'},
            'burkholderia mallei': {'type': 'bacteria', 'disease': 'Glanders', 'bsl': '3'},
            'burkholderia pseudomallei': {'type': 'bacteria', 'disease': 'Melioidosis', 'bsl': '3'},
            'clostridium botulinum': {'type': 'bacteria', 'disease': 'Botulism', 'bsl': '2/3'},
            'francisella tularensis': {'type': 'bacteria', 'disease': 'Tularemia', 'bsl': '2/3'},
            'yersinia pestis': {'type': 'bacteria', 'disease': 'Plague', 'bsl': '2/3'},
            
            # Viruses
            'ebola virus': {'type': 'virus', 'disease': 'Ebola hemorrhagic fever', 'bsl': '4'},
            'marburg virus': {'type': 'virus', 'disease': 'Marburg hemorrhagic fever', 'bsl': '4'},
            'lassa fever virus': {'type': 'virus', 'disease': 'Lassa fever', 'bsl': '4'},
            'crimean-congo hemorrhagic fever virus': {'type': 'virus', 'disease': 'CCHF', 'bsl': '4'},
            'rift valley fever virus': {'type': 'virus', 'disease': 'Rift Valley fever', 'bsl': '3/4'},
            'hendra virus': {'type': 'virus', 'disease': 'Hendra virus infection', 'bsl': '4'},
            'nipah virus': {'type': 'virus', 'disease': 'Nipah virus infection', 'bsl': '4'},
            'variola virus': {'type': 'virus', 'disease': 'Smallpox', 'bsl': '4'},
            
            # Fungi
            'coccidioides immitis': {'type': 'fungus', 'disease': 'Coccidioidomycosis', 'bsl': '2/3'},
            'coccidioides posadasii': {'type': 'fungus', 'disease': 'Coccidioidomycosis', 'bsl': '2/3'},
            
            # Toxins
            'ricin': {'type': 'toxin', 'source': 'Ricinus communis', 'ld50': '1-20 mg/kg'},
            'saxitoxin': {'type': 'toxin', 'source': 'Marine dinoflagellates', 'ld50': '0.01 mg/kg'},
            'tetrodotoxin': {'type': 'toxin', 'source': 'Pufferfish, blue-ringed octopus', 'ld50': '0.01 mg/kg'},
            'botulinum toxin': {'type': 'toxin', 'source': 'Clostridium botulinum', 'ld50': '0.001 mg/kg'},
            'staphylococcal enterotoxin b': {'type': 'toxin', 'source': 'Staphylococcus aureus', 'notes': 'SEB'},
            't-2 toxin': {'type': 'toxin', 'source': 'Fusarium species', 'notes': 'Trichothecene mycotoxin'},
        }
        
        # USDA Select Agents (plants and livestock)
        self.usda_select_agents = {
            'african swine fever virus': {'type': 'virus', 'host': 'Swine', 'bsl': '3'},
            'avian influenza h5n1': {'type': 'virus', 'host': 'Poultry', 'bsl': '3'},
            'foot-and-mouth disease virus': {'type': 'virus', 'host': 'Livestock', 'bsl': '3'},
            'rinderpest virus': {'type': 'virus', 'host': 'Ruminants', 'bsl': '3'},
        }
        
        # Alternative names and synonyms
        self.agent_aliases = {
            'anthrax': 'bacillus anthracis',
            'plague': 'yersinia pestis',
            'tularemia': 'francisella tularensis',
            'glanders': 'burkholderia mallei',
            'melioidosis': 'burkholderia pseudomallei',
            'smallpox': 'variola virus',
            'seb': 'staphylococcal enterotoxin b',
            'botox': 'botulinum toxin',
            'h5n1': 'avian influenza h5n1',
        }
    
    def check_select_agent_status(self, agent_name: str) -> Optional[DatabaseResult]:
        """Check if biological agent is on the CDC Select Agents list"""
        
        if not agent_name or len(agent_name.strip()) < 2:
            return None
        
        # Normalize input
        agent_lower = agent_name.lower().strip()
        
        # Check aliases
        if agent_lower in self.agent_aliases:
            agent_lower = self.agent_aliases[agent_lower]
        
        # Check HHS Select Agents
        for agent, details in self.hhs_select_agents.items():
            if self._agent_matches(agent_lower, agent):
                return DatabaseResult(
                    entity=agent_name,
                    database="CDC Select Agents (HHS)",
                    risk_level="critical_risk",
                    confidence=0.95,
                    category=f"Select Agent ({details['type'].title()})",
                    restrictions=f"Requires registration, BSL-{details.get('bsl', '3')} containment minimum",
                    regulatory_status="42 CFR Part 73 - Possession, use, and transfer restrictions apply",
                    explanation=f"HHS Select Agent: {agent} - {details.get('disease', 'See pathogen database')}",
                    source_url="https://www.selectagents.gov/"
                )
        
        # Check USDA Select Agents
        for agent, details in self.usda_select_agents.items():
            if self._agent_matches(agent_lower, agent):
                return DatabaseResult(
                    entity=agent_name,
                    database="CDC Select Agents (USDA)",
                    risk_level="high_risk",
                    confidence=0.90,
                    category=f"Agricultural Select Agent ({details['type'].title()})",
                    restrictions=f"Requires USDA registration, BSL-{details.get('bsl', '3')} containment",
                    regulatory_status="9 CFR Part 121 - Agricultural pathogen restrictions",
                    explanation=f"USDA Select Agent: {agent} - affects {details.get('host', 'livestock')}",
                    source_url="https://www.selectagents.gov/"
                )
        
        return None
    
    def _agent_matches(self, input_name: str, db_name: str) -> bool:
        """Check if agent names match with fuzzy matching for biological nomenclature"""
        # Exact match
        if input_name == db_name:
            return True
        
        # Check if input contains the database name or vice versa
        if db_name in input_name or input_name in db_name:
            return True
        
        # For scientific names, check genus/species separately
        if ' ' in db_name:
            genus, species = db_name.split()[0], db_name.split()[1]
            if genus in input_name and species in input_name:
                return True
        
        return False


class DEAControlledSubstancesDatabase:
    """
    DEA Controlled Substances and Precursor Chemicals Database
    
    Maintains schedules of controlled substances and regulated precursor chemicals
    under the Controlled Substances Act (CSA) and related regulations.
    """
    
    def __init__(self):
        self.last_updated = datetime.now()
        
        # Schedule I: High abuse potential, no accepted medical use
        self.schedule_1_substances = {
            'heroin': {'dea_number': '9200', 'chemical_name': 'Diacetylmorphine'},
            'lsd': {'dea_number': '7315', 'chemical_name': 'Lysergic acid diethylamide'},
            'mdma': {'dea_number': '7405', 'chemical_name': '3,4-Methylenedioxymethamphetamine'},
            'pcp': {'dea_number': '7471', 'chemical_name': 'Phencyclidine'},
            'thc': {'dea_number': '7370', 'chemical_name': 'Tetrahydrocannabinol'},
        }
        
        # List I Chemicals: Precursors for controlled substances
        self.list_1_chemicals = {
            'pseudoephedrine': {
                'threshold': '9 grams',
                'use': 'Methamphetamine precursor',
                'regulation': '21 CFR 1310'
            },
            'ephedrine': {
                'threshold': '9 grams', 
                'use': 'Methamphetamine precursor',
                'regulation': '21 CFR 1310'
            },
            'phenylpropanolamine': {
                'threshold': '9 grams',
                'use': 'Methamphetamine precursor', 
                'regulation': '21 CFR 1310'
            },
            'methylamine': {
                'threshold': '16 ounces',
                'use': 'Methamphetamine precursor',
                'regulation': '21 CFR 1310'
            },
            'phenylacetic acid': {
                'threshold': '16 ounces',
                'use': 'P2P methamphetamine precursor',
                'regulation': '21 CFR 1310'
            },
            'benzyl cyanide': {
                'threshold': '16 ounces', 
                'use': 'P2P methamphetamine precursor',
                'regulation': '21 CFR 1310'
            },
        }
        
        # List II Chemicals: Solvents, reagents, and other chemicals
        self.list_2_chemicals = {
            'acetone': {
                'threshold': '50 gallons',
                'use': 'Cocaine processing solvent',
                'regulation': '21 CFR 1310'
            },
            'toluene': {
                'threshold': '50 gallons',
                'use': 'Cocaine processing solvent', 
                'regulation': '21 CFR 1310'
            },
            'methyl ethyl ketone': {
                'threshold': '50 gallons',
                'use': 'Cocaine processing solvent',
                'regulation': '21 CFR 1310'
            },
            'hydrochloric acid': {
                'threshold': '30 gallons',
                'use': 'Drug synthesis reagent',
                'regulation': '21 CFR 1310'
            },
            'sulfuric acid': {
                'threshold': '35 gallons',
                'use': 'Drug synthesis reagent',
                'regulation': '21 CFR 1310'
            },
        }
        
        # Alternative names and synonyms
        self.substance_aliases = {
            'meth': 'methamphetamine',
            'crystal meth': 'methamphetamine', 
            'ecstasy': 'mdma',
            'molly': 'mdma',
            'angel dust': 'pcp',
            'acid': 'lsd',
            'sudafed': 'pseudoephedrine',
            'hcl': 'hydrochloric acid',
        }
    
    def check_controlled_status(self, substance_name: str) -> Optional[DatabaseResult]:
        """Check if substance is DEA controlled or regulated precursor"""
        
        if not substance_name or len(substance_name.strip()) < 2:
            return None
        
        # Normalize input
        substance_lower = substance_name.lower().strip()
        
        # Check aliases
        if substance_lower in self.substance_aliases:
            substance_lower = self.substance_aliases[substance_lower]
        
        # Check Schedule I substances
        for substance, details in self.schedule_1_substances.items():
            if self._substance_matches(substance_lower, substance):
                return DatabaseResult(
                    entity=substance_name,
                    database="DEA Schedule I",
                    risk_level="critical_risk",
                    confidence=0.95,
                    category="Schedule I Controlled Substance",
                    restrictions="Manufacture, distribution, and possession prohibited except for approved research",
                    regulatory_status="CSA Schedule I - No accepted medical use, high abuse potential",
                    explanation=f"Schedule I substance: {details['chemical_name']} (DEA #{details['dea_number']})",
                    source_url="https://www.deadiversion.usdoj.gov/schedules/"
                )
        
        # Check List I chemicals (direct precursors)
        for chemical, details in self.list_1_chemicals.items():
            if self._substance_matches(substance_lower, chemical):
                return DatabaseResult(
                    entity=substance_name,
                    database="DEA List I Chemical",
                    risk_level="high_risk", 
                    confidence=0.85,
                    category="Controlled Precursor Chemical",
                    restrictions=f"Regulated above {details['threshold']}, registration required for transactions",
                    regulatory_status=f"{details['regulation']} - Chemical precursor regulations apply",
                    explanation=f"List I precursor chemical: {chemical} - {details['use']}",
                    source_url="https://www.deadiversion.usdoj.gov/chemicals/"
                )
        
        # Check List II chemicals (processing chemicals)
        for chemical, details in self.list_2_chemicals.items():
            if self._substance_matches(substance_lower, chemical):
                return DatabaseResult(
                    entity=substance_name,
                    database="DEA List II Chemical",
                    risk_level="medium_risk",
                    confidence=0.75,
                    category="Regulated Processing Chemical",
                    restrictions=f"Regulated above {details['threshold']}, transaction reporting required",
                    regulatory_status=f"{details['regulation']} - Chemical transaction monitoring",
                    explanation=f"List II processing chemical: {chemical} - {details['use']}",
                    source_url="https://www.deadiversion.usdoj.gov/chemicals/"
                )
        
        return None
    
    def _substance_matches(self, input_name: str, db_name: str) -> bool:
        """Check if substance names match"""
        # Exact match
        if input_name == db_name:
            return True
        
        # Check if input contains the database name or vice versa
        if db_name in input_name or input_name in db_name:
            return True
        
        return False


class Phase1DatabaseIntegrator:
    """
    Phase 1 Database Integration Manager
    
    Coordinates queries across the three core safety databases:
    - CWC Schedules (chemical weapons)
    - CDC Select Agents (biological threats)
    - DEA Controlled Substances (drug precursors)
    """
    
    def __init__(self):
        self.cwc_db = CWCScheduleDatabase()
        self.cdc_db = CDCSelectAgentsDatabase()
        self.dea_db = DEAControlledSubstancesDatabase()
        
        self.databases = {
            'cwc': self.cwc_db,
            'cdc': self.cdc_db, 
            'dea': self.dea_db
        }
        
        self.query_count = 0
        self.cache = {}
        self.cache_expiry = timedelta(hours=24)
    
    async def enhanced_threat_assessment(self, entities: List[str]) -> List[DatabaseResult]:
        """
        Perform threat assessment across all Phase 1 databases
        
        Args:
            entities: List of chemical/biological entity names extracted from text
            
        Returns:
            List of DatabaseResult objects for any matches found
        """
        results = []
        
        for entity in entities:
            # Check cache first
            cache_key = entity.lower().strip()
            if cache_key in self.cache:
                cached_result, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < self.cache_expiry:
                    if cached_result:
                        results.append(cached_result)
                    continue
            
            # Query databases in priority order
            entity_results = await self._query_all_databases(entity)
            
            # Cache the result (including None results to avoid repeated queries)
            highest_risk = self._get_highest_risk_result(entity_results)
            self.cache[cache_key] = (highest_risk, datetime.now())
            
            if highest_risk:
                results.append(highest_risk)
                
        return results
    
    async def _query_all_databases(self, entity: str) -> List[DatabaseResult]:
        """Query all databases for a single entity"""
        results = []
        
        # CWC Chemical Weapons (highest priority)
        if cwc_result := self.cwc_db.check_cwc_status(entity):
            results.append(cwc_result)
            
        # CDC Select Agents (biological threats)
        if cdc_result := self.cdc_db.check_select_agent_status(entity):
            results.append(cdc_result)
            
        # DEA Controlled Substances (drug precursors)
        if dea_result := self.dea_db.check_controlled_status(entity):
            results.append(dea_result)
        
        self.query_count += 1
        return results
    
    def _get_highest_risk_result(self, results: List[DatabaseResult]) -> Optional[DatabaseResult]:
        """Return the highest risk result from multiple database matches"""
        if not results:
            return None
        
        risk_priority = {
            'critical_risk': 4,
            'high_risk': 3, 
            'medium_risk': 2,
            'low_risk': 1
        }
        
        return max(results, key=lambda x: risk_priority.get(x.risk_level, 0))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            'total_queries': self.query_count,
            'cache_size': len(self.cache),
            'databases_active': len(self.databases),
            'last_updated': min([
                self.cwc_db.last_updated,
                self.cdc_db.last_updated,
                self.dea_db.last_updated
            ]).isoformat()
        }


# Example usage and testing
if __name__ == "__main__":
    async def test_phase1_integration():
        """Test the Phase 1 database integration"""
        integrator = Phase1DatabaseIntegrator()
        
        # Test cases covering different database domains
        test_entities = [
            # Chemical weapons (CWC)
            'sarin',
            'mustard gas',
            'phosphorus oxychloride',
            
            # Biological agents (CDC)
            'bacillus anthracis',
            'ebola virus', 
            'ricin',
            
            # Drug precursors (DEA)
            'pseudoephedrine',
            'acetone',
            'methylamine',
            
            # Benign chemicals (should not match)
            'sodium chloride',
            'water',
            'aspirin'
        ]
        
        print("🧪 Phase 1 Database Integration Test")
        print("=" * 50)
        
        results = await integrator.enhanced_threat_assessment(test_entities)
        
        for result in results:
            print(f"\n🚨 THREAT DETECTED: {result.entity}")
            print(f"   Database: {result.database}")
            print(f"   Risk Level: {result.risk_level.upper()}")
            print(f"   Category: {result.category}")
            print(f"   Explanation: {result.explanation}")
            print(f"   Restrictions: {result.restrictions}")
        
        print(f"\n📊 Statistics: {integrator.get_statistics()}")
    
    # Run the test
    asyncio.run(test_phase1_integration())
