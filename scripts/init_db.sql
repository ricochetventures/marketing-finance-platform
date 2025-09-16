-- scripts/init_db.sql
-- Create tables for the marketing-finance platform

CREATE SCHEMA IF NOT EXISTS marketing;

-- Companies table
CREATE TABLE IF NOT EXISTS marketing.companies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    ticker VARCHAR(10),
    industry VARCHAR(100),
    market_cap DECIMAL(20, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agencies table
CREATE TABLE IF NOT EXISTS marketing.agencies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    holding_company VARCHAR(255),
    type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Marketing spend table
CREATE TABLE IF NOT EXISTS marketing.spend_data (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES marketing.companies(id),
    date DATE NOT NULL,
    total_spend DECIMAL(15, 2),
    digital_spend DECIMAL(15, 2),
    tv_spend DECIMAL(15, 2),
    print_spend DECIMAL(15, 2),
    other_spend DECIMAL(15, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stock prices table
CREATE TABLE IF NOT EXISTS marketing.stock_prices (
    id SERIAL PRIMARY KEY,
    company_id INTEGER REFERENCES marketing.companies(id),
    date DATE NOT NULL,
    open_price DECIMAL(10, 2),
    close_price DECIMAL(10, 2),
    high_price DECIMAL(10, 2),
    low_price DECIMAL(10, 2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_spend_company_date ON marketing.spend_data(company_id, date);
CREATE INDEX idx_stock_company_date ON marketing.stock_prices(company_id, date);

-- Create materialized view for performance
CREATE MATERIALIZED VIEW IF NOT EXISTS marketing.company_performance AS
SELECT 
    c.id,
    c.name,
    c.industry,
    AVG(s.total_spend) as avg_spend,
    AVG(sp.close_price) as avg_stock_price
FROM marketing.companies c
LEFT JOIN marketing.spend_data s ON c.id = s.company_id
LEFT JOIN marketing.stock_prices sp ON c.id = sp.company_id
GROUP BY c.id, c.name, c.industry;