"""
Comprehensive agency intelligence database
"""

AGENCY_PROFILES = {
    'WPP': {
        'full_name': 'WPP plc',
        'headquarters': 'London, UK',
        'revenue_2023': '$17.1B',
        'employees': '109,000+',
        'holding_company': 'WPP',
        
        'key_agencies': [
            'Ogilvy', 'GroupM', 'VMLY&R', 'Wunderman Thompson', 'Grey', 'Mindshare', 'MediaCom'
        ],
        
        'sector_expertise': {
            'Healthcare/Pharma': {
                'strength': 9.0,
                'specialties': ['Patient engagement', 'HCP marketing', 'Clinical trial recruitment'],
                'key_clients': ['GSK', 'Pfizer', 'AstraZeneca'],
                'case_studies': 'Unbranded disease awareness campaigns, Patient support programs'
            },
            'Beauty & Personal Care': {
                'strength': 9.5,
                'specialties': ['Influencer marketing', 'E-commerce optimization', 'Brand positioning'],
                'key_clients': ['Unilever', 'P&G', 'L\'Oréal'],
                'case_studies': 'Dove Real Beauty campaign, Olay Skin Advisor AI tool'
            },
            'Technology': {
                'strength': 8.5,
                'specialties': ['B2B marketing', 'Product launches', 'Thought leadership'],
                'key_clients': ['Google', 'Microsoft', 'Intel'],
                'case_studies': 'Google Year in Search, Microsoft Azure campaigns'
            },
            'Beverages': {
                'strength': 8.0,
                'specialties': ['Mass market appeal', 'Sports marketing', 'Global campaigns'],
                'key_clients': ['Coca-Cola (historical)', 'Diageo'],
                'case_studies': 'Coke Zero Sugar repositioning'
            }
        },
        
        'capabilities': {
            'Media': {
                'rating': 9.5,
                'details': 'GroupM is world\'s largest media investment company. Advanced programmatic buying, TV optimization',
                'tools': ['Choreograph (data & analytics)', 'GroupM Nexus (tech platform)']
            },
            'Creative': {
                'rating': 9.0,
                'details': 'Ogilvy legacy of creative excellence. Strong in integrated campaigns',
                'awards': 'Cannes Lions Network of the Year 2022'
            },
            'Audience Strategy': {
                'rating': 9.0,
                'details': 'Advanced first-party data platforms, CDP integration expertise',
                'tools': ['Choreograph', 'mPlatform']
            },
            'CRM': {
                'rating': 8.5,
                'details': 'Wunderman Thompson specializes in customer experience and CRM',
                'platforms': ['Salesforce', 'Adobe', 'Oracle']
            },
            'Production': {
                'rating': 8.0,
                'details': 'Hogarth for production, The&Partnership for integration',
                'scale': 'Global production hubs in 35+ countries'
            },
            'Digital Commerce': {
                'rating': 9.0,
                'details': 'Acceleration unit for e-commerce, Salmon for commerce platforms',
                'expertise': ['Amazon strategy', 'DTC optimization', 'Marketplace management']
            }
        },
        
        'strengths': [
            'Global scale and consistency',
            'Data and analytics leadership',
            'Comprehensive service offering',
            'Strong holding company resources'
        ],
        
        'weaknesses': [
            'Can be bureaucratic at scale',
            'Integration challenges across agencies',
            'Premium pricing'
        ],
        
        'ideal_for': [
            'Large global brands seeking consistency',
            'Data-driven marketers',
            'Companies needing full-service solution'
        ]
    },
    
    'Publicis': {
        'full_name': 'Publicis Groupe',
        'headquarters': 'Paris, France',
        'revenue_2023': '$13.1B',
        'employees': '98,000+',
        'holding_company': 'Publicis',
        
        'key_agencies': [
            'Publicis Worldwide', 'Leo Burnett', 'Saatchi & Saatchi', 'Zenith', 'Starcom', 'Sapient'
        ],
        
        'sector_expertise': {
            'Healthcare/Pharma': {
                'strength': 9.5,
                'specialties': ['Oncology marketing', 'Rare disease campaigns', 'Digital health'],
                'key_clients': ['Novartis', 'Sanofi', 'Roche'],
                'case_studies': 'Virtual clinical trials, Patient finder programs'
            },
            'Beauty & Personal Care': {
                'strength': 9.0,
                'specialties': ['Luxury beauty', 'Social commerce', 'Influencer networks'],
                'key_clients': ['L\'Oréal (major partnership)', 'Estée Lauder'],
                'case_studies': 'L\'Oréal Virtual Try-On, Makeup Genius app'
            },
            'Technology': {
                'strength': 9.0,
                'specialties': ['Digital transformation', 'Cloud marketing', 'SaaS positioning'],
                'key_clients': ['Samsung', 'Hewlett Packard Enterprise'],
                'case_studies': 'Samsung Galaxy launches, HPE hybrid cloud campaigns'
            },
            'Automotive': {
                'strength': 8.5,
                'specialties': ['EV marketing', 'Dealer network support', 'Test drive programs'],
                'key_clients': ['Stellantis', 'Renault'],
                'case_studies': 'Jeep brand repositioning, Peugeot electric vehicle launches'
            }
        },
        
        'capabilities': {
            'Media': {
                'rating': 9.0,
                'details': 'Zenith and Starcom for media. Publicis Media leads in digital innovation',
                'tools': ['Marcel (AI platform)', 'Epsilon PeopleCloud']
            },
            'Creative': {
                'rating': 8.5,
                'details': 'Leo Burnett for heartland brands, Publicis for modern brands',
                'awards': 'Strong Cannes performance, especially in digital/innovation'
            },
            'Audience Strategy': {
                'rating': 10.0,
                'details': 'Industry-leading with Epsilon acquisition (largest first-party data set)',
                'tools': ['Epsilon Core ID', 'CORE (customer data)', 'Conversant (identity)']
            },
            'CRM': {
                'rating': 9.5,
                'details': 'Epsilon specializes in data-driven CRM and loyalty',
                'scale': '200M+ consumer profiles in US alone'
            },
            'Production': {
                'rating': 8.5,
                'details': 'Prodigious for production at scale, Fallon for premium',
                'capabilities': ['In-house studios', 'Real-time content creation']
            },
            'Digital Transformation': {
                'rating': 9.5,
                'details': 'Sapient for technology consulting and digital transformation',
                'expertise': ['Cloud migration', 'API development', 'Platform integration']
            }
        },
        
        'strengths': [
            'Strongest first-party data capabilities (Epsilon)',
            'Digital transformation expertise (Sapient)',
            'AI and machine learning leadership',
            'Performance marketing excellence'
        ],
        
        'weaknesses': [
            'Creative sometimes seen as secondary to data',
            'Complex organizational structure post-mergers',
            'US-heavy data assets (less global coverage)'
        ],
        
        'ideal_for': [
            'Brands prioritizing data-driven marketing',
            'Companies undergoing digital transformation',
            'Performance-focused marketers',
            'Retail and e-commerce heavy industries'
        ]
    },
    
    'Omnicom': {
        'full_name': 'Omnicom Group',
        'headquarters': 'New York, USA',
        'revenue_2023': '$14.3B',
        'employees': '70,000+',
        'holding_company': 'Omnicom',
        
        'key_agencies': [
            'BBDO', 'DDB', 'TBWA', 'OMD', 'PHD', 'FleishmanHillard'
        ],
        
        'sector_expertise': {
            'Beverages': {
                'strength': 9.5,
                'specialties': ['Brand building', 'Sports marketing', 'Cultural moments'],
                'key_clients': ['PepsiCo', 'Anheuser-Busch InBev', 'Diageo'],
                'case_studies': 'Pepsi Super Bowl campaigns, Bud Light sports partnerships'
            },
            'Automotive': {
                'strength': 9.0,
                'specialties': ['Launch campaigns', 'Dealer marketing', 'Customer journey'],
                'key_clients': ['Volkswagen', 'Nissan', 'FCA'],
                'case_studies': 'VW "Drive Bigger" campaign, Nissan Innovation series'
            },
            'Financial Services': {
                'strength': 8.5,
                'specialties': ['Trust building', 'Product launches', 'Regulatory compliance'],
                'key_clients': ['American Express', 'Visa'],
                'case_studies': 'Amex Small Business Saturday'
            },
            'Technology': {
                'strength': 8.0,
                'specialties': ['Consumer tech', 'Innovation narratives', 'Product storytelling'],
                'key_clients': ['Apple (historical)', 'AT&T'],
                'case_studies': 'Apple "Think Different" legacy'
            }
        },
        
        'capabilities': {
            'Media': {
                'rating': 9.0,
                'details': 'OMD and PHD are precision media specialists. Strong in innovation',
                'tools': ['Omni (planning platform)', 'Annalect (data & analytics)']
            },
            'Creative': {
                'rating': 10.0,
                'details': 'BBDO and TBWA are creative powerhouses. DDB for challenger brands',
                'awards': 'Most awarded network at Cannes Lions historically',
                'philosophy': '"The Work, The Work, The Work" - creative excellence first'
            },
            'Audience Strategy': {
                'rating': 8.0,
                'details': 'Annalect for data strategy, growing but not Epsilon-level',
                'tools': ['Omni ID graph', 'Annalect Data Cloud']
            },
            'CRM': {
                'rating': 7.5,
                'details': 'Merkle acquisition attempts fell through. Uses partners',
                'approach': 'Best-of-breed partnerships rather than owned platform'
            },
            'Production': {
                'rating': 8.5,
                'details': 'Omnicom Production Group, plus in-house at major agencies',
                'scale': 'Content studios in major markets'
            },
            'Brand Strategy': {
                'rating': 9.5,
                'details': 'Exceptional brand positioning and strategy work',
                'expertise': ['Brand architecture', 'Positioning', 'Cultural insight']
            }
        },
        
        'strengths': [
            'Unmatched creative excellence',
            'Strong agency brands (BBDO, TBWA have independent equity)',
            'Excellent at big brand building campaigns',
            'Cultural relevance and creativity'
        ],
        
        'weaknesses': [
            'Less advanced in data/tech vs Publicis',
            'Decentralized structure can create silos',
            'CRM capabilities lag competitors'
        ],
        
        'ideal_for': [
            'Brands prioritizing creative excellence',
            'Companies with big brand-building budgets',
            'Advertisers in creative-sensitive categories',
            'Marketers valuing cultural impact over performance metrics'
        ]
    },
    
    # Add more agencies following same pattern...
    'IPG': {
        'full_name': 'Interpublic Group',
        'headquarters': 'New York, USA',
        'revenue_2023': '$10.9B',
        'employees': '54,000+',
        
        'sector_expertise': {
            'Healthcare/Pharma': {
                'strength': 9.0,
                'specialties': ['IPG Health network', 'Patient centricity', 'Medical education'],
                'key_clients': ['Johnson & Johnson', 'Pfizer'],
                'case_studies': 'Leading pharma-specific network'
            }
        },
        
        'capabilities': {
            'Media': {'rating': 8.5, 'details': 'UM and Initiative for media'},
            'Creative': {'rating': 8.0, 'details': 'McCann, FCB for creative'},
            'Healthcare': {'rating': 9.5, 'details': 'Industry-leading healthcare network'},
            'Data': {'rating': 8.5, 'details': 'Acxiom partnership for data'},
        },
        
        'ideal_for': ['Healthcare marketers', 'Mid-sized brands', 'Value-conscious clients']
    }
}

def get_agency_recommendation(company_name: str, industry: str, priorities: dict) -> dict:
    """
    AI-powered agency recommendation based on company needs
    
    priorities = {
        'creative_importance': 0-10,
        'data_importance': 0-10,
        'global_scale_needed': bool,
        'budget_sensitivity': 'low'|'medium'|'high'
    }
    """
    
    scores = {}
    
    for agency_name, profile in AGENCY_PROFILES.items():
        score = 0
        reasoning = []
        
        # Sector expertise match
        if industry in profile.get('sector_expertise', {}):
            sector = profile['sector_expertise'][industry]
            score += sector['strength'] * 10
            reasoning.append(f"Strong {industry} expertise (rated {sector['strength']}/10)")
        
        # Creative vs Data priorities
        creative_rating = profile.get('capabilities', {}).get('Creative', {}).get('rating', 5)
        data_rating = profile.get('capabilities', {}).get('Audience Strategy', {}).get('rating', 5)
        
        creative_weight = priorities.get('creative_importance', 5) / 10
        data_weight = priorities.get('data_importance', 5) / 10
        
        score += (creative_rating * creative_weight + data_rating * data_weight) * 5
        
        # Global scale
        if priorities.get('global_scale_needed', False):
            if int(profile.get('employees', '0+').replace('+', '').replace(',', '')) > 80000:
                score += 15
                reasoning.append("Global scale and presence")
        
        scores[agency_name] = {
            'score': score,
            'reasoning': reasoning,
            'profile': profile
        }
    
    # Rank by score
    ranked = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    return {
        'top_recommendation': ranked[0][0],
        'alternatives': [r[0] for r in ranked[1:3]],
        'analysis': ranked[0][1]
    }