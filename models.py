from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    role = db.Column(db.String(20), default='doctor')  # Options: admin, doctor, nurse

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10))
    email = db.Column(db.String(120))
    phone = db.Column(db.String(20))
    address = db.Column(db.String(200))
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(),
                           onupdate=db.func.current_timestamp())

    # Define relationship with doctors (users)
    doctor = db.relationship('User', backref=db.backref('patients', lazy=True))

    # Define relationship with health records
    health_records = db.relationship('HealthRecord', backref='patient', lazy=True,
                                     cascade="all, delete-orphan")

    def get_full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __repr__(self):
        return f"<Patient {self.get_full_name()}>"


class HealthRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patient.id'), nullable=False)
    record_date = db.Column(db.DateTime, default=db.func.current_timestamp())
    pregnancies = db.Column(db.Integer)
    glucose = db.Column(db.Float)
    blood_pressure = db.Column(db.Float)
    skin_thickness = db.Column(db.Float)
    insulin = db.Column(db.Float)
    bmi = db.Column(db.Float)
    diabetes_pedigree_function = db.Column(db.Float)
    age = db.Column(db.Integer)
    risk_probability = db.Column(db.Float)
    risk_category = db.Column(db.String(20))
    notes = db.Column(db.Text)

    def __repr__(self):
        return f"<HealthRecord for Patient {self.patient_id} on {self.record_date}>"