"""
Database Models & ORM Schema Module
====================================
Domain: Healthcare / Computer Vision / Deep Learning
Dataset: APTOS 2019 Blindness Detection
Task: Defines persistent database entities for Patients, Predictions, XAI Heatmaps, and PDF Reports.

Author: Senior Software Architect & MLOps Engineer
"""

import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("PredictionRecord", back_populates="patient")


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String(50), ForeignKey("patients.patient_id"), nullable=False)
    filename = Column(String(255), nullable=False)
    predicted_class = Column(String(50), nullable=False)
    predicted_index = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    recommendation = Column(Text, nullable=False)
    class_probabilities_json = Column(Text, nullable=False)
    layer_name = Column(String(100), default="top_activation")
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")
    report = relationship("ClinicalReportRecord", back_populates="prediction", uselist=False)


class ClinicalReportRecord(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    pdf_filename = Column(String(255), nullable=False)
    pdf_filepath = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    prediction = relationship("PredictionRecord", back_populates="report")
